import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.dual_path import *
from src.models.CSE_transformer import SBTransformerBlock_CSE

class Sepformer(nn.Module):
    def __init__(self, num_spks=2, add_mt=False, ctx_dim=4096, ce=True) -> None:
        super().__init__()
        self.encoder = Encoder(kernel_size=16, out_channels=256)
        self.masknet = Dual_Path_Model_CSE(
            num_spks=num_spks,
            in_channels=256,
            out_channels=256,
            num_layers=2,
            K=250,
            intra_model=SBTransformerBlock_CSE(
                num_layers=8,
                d_model=256,
                nhead=8,
                d_ffn=1024,
                dropout=0,
                use_positional_encoding=True,
                norm_before=True,
            ),
            inter_model=SBTransformerBlock_CSE(
                num_layers=8,
                d_model=256,
                nhead=8,
                d_ffn=1024,
                dropout=0,
                use_positional_encoding=True,
                norm_before=True,
            ),
            norm="ln",
            linear_layer_after_inter_intra=False,
            skip_around_intra=True,
            llm_dim=ctx_dim if add_mt else None, ##ADD
        )
        self.decoder = Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)
        self.context_selector = None
        self.num_spks = num_spks
        self.add_ctx = add_mt
        self.ce = ce

    def add_mt_pipeline(self):
        self.masknet.add_ctx()
        if self.num_spks == 2 and not self.ce:
            self.context_selector = nn.Linear(256, 1)
        else:
            self.context_selector = nn.Linear(256, self.num_spks)

    def forward(self, mix: torch.Tensor, ctx: torch.Tensor, se=None) -> torch.Tensor:
        """Run extraction on batch of audio.

        Arguments
        ---------
        mix : torch.Tensor
            The mixture of sources.

        Returns
        -------
        tensor
            Extracted audio
        """
        # mix : B x T

        ## Separation
        mix_w = self.encoder(mix)
        # mix_w : B x C x L (C: filter size)

        if not self.add_ctx:
            est_mask = self.masknet(mix_w, None)
        else: 
            est_mask, pred_head = self.masknet(mix_w, ctx)
            context_pred = self.context_selector(pred_head)
            # est_mask : Num_speaker x B x C X L
            
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        ## Decoding
        est_source = torch.cat(
            [self.decoder(sep_h[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )
        # est_source : B x T x Num_speaker

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        
        if not self.add_ctx:
            return est_source
        else:
            return est_source, context_pred


class Dual_Path_Model_CSE(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
        llm_dim=4096, ##ADD
    ):
        super(Dual_Path_Model_CSE, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block_CSE(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                        llm_dim=llm_dim, ##ADD
                    )
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def add_ctx(self):
        for i in range(self.num_layers):
            self.dual_mdl[i].add_ctx()

    def forward(self, x, ctx):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                x, pred_head = self.dual_mdl[i](x, ctx)
            else:
                x, _ = self.dual_mdl[i](x, ctx)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x, pred_head

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = (
                torch.Tensor(torch.zeros(B, N, gap))
                .type(input.dtype)
                .to(input.device)
            )
            input = torch.cat([input, pad], dim=2)

        _pad = (
            torch.Tensor(torch.zeros(B, N, P))
            .type(input.dtype)
            .to(input.device)
        )
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input
    
class Dual_Computation_Block_CSE(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        llm_dim=4096, ##ADD
    ):
        super(Dual_Computation_Block_CSE, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra
        self.llm_dim = llm_dim
        self.out_channels = out_channels

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            if isinstance(intra_mdl, SBRNNBlock):
                self.intra_linear = Linear(
                    out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
                )
            else:
                self.intra_linear = Linear(
                    out_channels, input_size=out_channels
                )

            if isinstance(inter_mdl, SBRNNBlock):
                self.inter_linear = Linear(
                    out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
                )
            else:
                self.inter_linear = Linear(
                    out_channels, input_size=out_channels
                )

        self.intra_context_mapper = None
        self.inter_context_mapper = None

    def add_ctx(self):
        self.intra_context_mapper = nn.Linear(self.llm_dim, self.out_channels) ##ADD
        self.inter_context_mapper = nn.Linear(self.llm_dim, self.out_channels) ##ADD

    def forward(self, x, ctx):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        ## Context is appended as prompt for each chunk (at the front timestep)
        if ctx is not None:
            ctx_length = ctx.size(1)    # B, T, C
            intra_c = self.intra_context_mapper(ctx) ##ADD
            intra_c = intra_c.unsqueeze(1).repeat(1, S, 1, 1).view(B * S, ctx_length, -1) ##ADD
            intra = torch.cat([intra_c, intra], dim=1) ##ADD
        else:
            ctx_length = 0
        intra = self.intra_mdl(intra)
        ## Detach context part
        intra = intra[:, ctx_length:] ##ADD

        # [BS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        
        # [BK, S, H]
        ## Context is appended as prompt for across chunk (global; at the front timestep)
        if ctx is not None:
            inter_c = self.inter_context_mapper(ctx) ##ADD
            inter_c = inter_c.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, ctx_length, -1) ##ADD
            inter = torch.cat([inter_c, inter], 1) ##ADD
        inter = self.inter_mdl(inter)
        ## Detach context part
        pred_head = inter[:, 0] ##ADD
        pred_head = pred_head.view(B, K, -1).mean(1)
        inter = inter[:, ctx_length:] ##ADD

        # [BK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out, pred_head