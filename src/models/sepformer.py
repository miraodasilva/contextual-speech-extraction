import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.dual_path import Decoder, Dual_Path_Model, Encoder
from src.models.CSE_transformer import SBTransformerBlock_CSE

class Sepformer(nn.Module):
    def __init__(self, num_spks=2) -> None:
        super().__init__()
        self.encoder = Encoder(kernel_size=16, out_channels=256)
        self.masknet = Dual_Path_Model(
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
        )
        self.decoder = Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)
        self.num_spks = num_spks

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
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

        est_mask = self.masknet(mix_w)
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
        return est_source
