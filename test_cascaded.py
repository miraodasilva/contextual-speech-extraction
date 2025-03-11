import argparse
import random
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.distributed as dist
import soundfile as sf
import contextlib
import torchmetrics
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
import torchaudio

# model
import whisper
from transformers import LlamaForCausalLM
from src.models.sepformer import Sepformer
# data
from src.data.dataset_train_CSE import CSEDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dailytalk_data_path', default='dir_to/DailyTalk_processed')
    parser.add_argument('--spokenwoz_data_path', default='dir_to/SpokenWoz_processed')
    parser.add_argument('--tedlium_data_path', default='dir_to/TEDLIUM_processed')
    parser.add_argument("--acoustic_noise_path", default='dir_to/DEMAND')

    parser.add_argument('--llama_path', default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--llama_auth_token', default='', help='specify the authorization token')

    parser.add_argument("--max_sp_len", type=int, default=16, help='max length in sec')
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--context_length", type=int, default=0, help='How many context will be used for evaulation; 0 uses full context')
    
    parser.add_argument("--num_max_mix", type=int, default=2, help='how many mixed speech will be used during training')
    parser.add_argument("--num_test_mix", type=int, default=2, help='Which evaluation set will be used 2 or 3 speaker mixed')
    parser.add_argument("--augmentation", default=False, action='store_true')
    parser.add_argument("--speed_perturb_ratio", type=str, default='0.9 1.0 1.1')
    parser.add_argument("--shift_prob", type=float, default=0.4)
    parser.add_argument("--max_shift_sec", type=float, default=1.)
    parser.add_argument("--max_context_train", type=int, default=100, help='What maximum context will be used for training')

    parser.add_argument("--test_dataset", type=str, default='dailytalk')

    parser.add_argument("--save_dir", type=str, default='./data/test_results')
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/Sepformer')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--project", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--update_frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--tot_iters", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup", default=False, action='store_true')
    parser.add_argument("--warmup_iteration", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.000001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=5000)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--mode", type=str, default='test', help='train, test, valid')
    parser.add_argument("--reset_optimizer", default=False, action='store_true')

    parser.add_argument("--fp16", default=False, action='store_true')

    parser.add_argument("--generate_speech", default=False, action='store_true', help='Whether saving audio during eval')
    parser.add_argument("--generate_step", type=int, default=1000)
    parser.add_argument("--num_gen_speech", type=int, default=20)

    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--masterport", type=str, default='1234')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args

def train_net(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = args.masterport

    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    
    assert args.mode == 'test'
    assert args.batch_size == 1

    args.save_dir = os.path.join(args.save_dir, *os.path.normpath(os.path.splitext(args.checkpoint)[0]).split(os.sep)[-2:])
    if not os.path.exists(args.save_dir) and args.local_rank == 0:
        os.makedirs(args.save_dir)
    
    args.speed_perturb_ratio = [float(ratio) for ratio in args.speed_perturb_ratio.split()]

    llm = LlamaForCausalLM.from_pretrained(args.llama_path, token=args.llama_auth_token, torch_dtype=torch.float16)
    llm.requires_grad_(False)
    llm.eval()

    model = Sepformer(num_spks=args.num_max_mix)
    asr = whisper.load_model("base")

    num_llm = sum(p.numel() for p in llm.parameters())
    num_model = sum(p.numel() for p in model.parameters())

    if args.local_rank == 0:
        print(f'Sepformer # of params: {num_model}')
        print(f'LLM # of params: {num_llm}')

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint

    llm.cuda()
    model.cuda()
    asr.cuda()

    if args.distributed:
        model = DDP(model, 
                    device_ids=[args.local_rank], 
                    output_device=args.local_rank, 
                    find_unused_parameters=False,
                    )

    _ = test(model, llm, asr, fast_validate=False, test_dataset=args.test_dataset)

def test(model, llm, asr, fast_validate=False, test_dataset=None):
    with torch.inference_mode():
        model.eval()
        asr.eval()

        val_data = CSEDataset(
            dailytalk_data_path=args.dailytalk_data_path,
            spokenwoz_data_path=args.spokenwoz_data_path, 
            tedlium_data_path=args.tedlium_data_path,
            llama_path=args.llama_path,
            mode='test',
            train_data=test_dataset, 
            max_sp_len=args.max_sp_len,
            context_length=args.context_length,
            auth_token=args.llama_auth_token,
            num_max_mix=args.num_max_mix,
            num_test_mix=args.num_test_mix,
            augmentation=args.augmentation,
            acoustic_noise_path=args.acoustic_noise_path,    
            speed_perturb_ratio=args.speed_perturb_ratio,
            max_shift_sec=args.max_shift_sec,
            shift_prob=args.shift_prob,
            max_context_train=args.max_context_train,
            sr=args.sr,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            collate_fn=lambda x: val_data.collate_fn_no_tok(x),
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(5 * batch_size, int(len(dataloader.dataset)))
            max_batches = 5
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        si_snr_criterion = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().cuda()
        sdr_criterion = torchmetrics.audio.SignalDistortionRatio().cuda()

        si_snr_prev_criterion = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().cuda()
        sdr_prev_criterion = torchmetrics.audio.SignalDistortionRatio().cuda()

        gts = []
        preds = []
        mixed = []
        sp_lens = []
        f_names = []


        dir_name = f'Cascaded_{args.num_test_mix}_speaker_{args.context_length}_ctx_{args.test_dataset}'

        description = 'Test on subset of the Val dataset' if fast_validate else 'Test'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 20 == 0:
                if not fast_validate:
                    print(f"******** Test ({test_dataset}) : %d / %d ********" % ((i + 1) * batch_size, samples))
            mixed_sp, gt_sp, context, sp_len, f_names = batch

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                # 1. separate
                candidates = rearrange(model(mixed_sp.cuda()), 'b t s -> b s t')  # B, T, num_sp -> B x num_sp x T
            with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                select_inds = []
                for b, candidate in enumerate(candidates):    # batch
                    context_inp = val_data.tokenizer(context[b], return_tensors="pt").input_ids
                    probs = []
                    for sample in candidate:    # speaker
                        # 2. ASR
                        sample = torchaudio.functional.resample(sample.unsqueeze(0), args.sr, 16_000).squeeze(0)
                        sample = sample / sample.abs().max() * 0.9
                        asr_pred = asr.transcribe(sample)['text'].lstrip()
                        
                        candidate_inp = val_data.tokenizer(asr_pred, return_tensors="pt").input_ids[:, 1:]  # eliminate bos
                        len_logit = candidate_inp.size(1)
                        context_input = torch.cat([context_inp, candidate_inp], 1)
                        # 3. LLM
                        logits = llm(input_ids=context_input.cuda()).logits # 1, T, C
                        prob = torch.nn.functional.log_softmax(logits[:, -len_logit:], -1).max(2)[0].sum() / len_logit
                        probs.append(prob)
                    select_ind = torch.argmax(torch.tensor(probs))
                    select_inds.append(select_ind)
                inds = torch.tensor(select_inds)   #B
                enhanced_sp = candidates[torch.arange(len(candidates)), inds]   #B,T

            si_snr_criterion.update(enhanced_sp.float(), gt_sp.cuda())
            sdr_criterion.update(enhanced_sp.float(), gt_sp.cuda())

            si_snr_prev_criterion.update(mixed_sp.cuda(), gt_sp.cuda())
            sdr_prev_criterion.update(mixed_sp.cuda(), gt_sp.cuda())

            if args.generate_speech:
                mixed = [sp for sp in mixed_sp]
                preds = [sp for sp in enhanced_sp.cpu()]
                gts = [sp for sp in gt_sp]
                sp_lens = [sp for sp in sp_len]
                if args.local_rank == 0:
                    for gt_sp, pred_sp, mixed_sp, sp_len, f_name in zip(gts, preds, mixed, sp_lens, f_names):
                        gt_save_name = os.path.join(args.save_dir, dir_name, f'audio_{test_dataset}', 'gts', f_name + '.wav')
                        pred_save_name = os.path.join(args.save_dir, dir_name, f'audio_{test_dataset}', 'preds', f_name + '.wav')
                        mixed_save_name = os.path.join(args.save_dir, dir_name, f'audio_{test_dataset}', 'mixed', f_name + '.wav')
                        if not os.path.exists(os.path.dirname(gt_save_name)):
                            os.makedirs(os.path.dirname(gt_save_name))
                        if not os.path.exists(os.path.dirname(pred_save_name)):
                            os.makedirs(os.path.dirname(pred_save_name))
                        if not os.path.exists(os.path.dirname(mixed_save_name)):
                            os.makedirs(os.path.dirname(mixed_save_name)) 

                        gt_sp = gt_sp.float().numpy()[:sp_len]
                        gt_sp = gt_sp / np.max(np.abs(gt_sp)) * 0.9
                        pred_sp = pred_sp.float().numpy()[:sp_len]
                        pred_sp = pred_sp / np.max(np.abs(pred_sp)) * 0.9
                        mixed_sp = mixed_sp.float().numpy()[:sp_len]
                        mixed_sp = mixed_sp / np.max(np.abs(mixed_sp)) * 0.9

                        sf.write(gt_save_name, gt_sp, samplerate=args.sr, subtype='PCM_16')
                        sf.write(pred_save_name, pred_sp, samplerate=args.sr, subtype='PCM_16')
                        sf.write(mixed_save_name, mixed_sp, samplerate=args.sr, subtype='PCM_16')

            if args.distributed:
                dist.barrier()

            if i >= max_batches:
                break

        val_sisnr = si_snr_criterion.compute()
        val_sdr = sdr_criterion.compute()
        val_sisnr_prev = si_snr_prev_criterion.compute()
        val_sdr_prev = sdr_prev_criterion.compute()
        if args.local_rank == 0:
            print("## Test SI-SNR: ", val_sisnr)
            print("## Test SDR: ", val_sdr)      
            print("## Test SI-SNR-i: ", val_sisnr - val_sisnr_prev)
            print("## Test SDR-i: ", val_sdr - val_sdr_prev)      

        if args.local_rank == 0:
            with open(os.path.join(args.save_dir, dir_name, f'test_results_{test_dataset}.txt'), 'w') as txt:
                txt.write(f'Test SI-SNR: {val_sisnr}\n')
                txt.write(f'Test SDR: {val_sdr}\n')
                txt.write(f'Test SI-SNR-i: {val_sisnr - val_sisnr_prev}\n')
                txt.write(f'Test SDR-i: {val_sdr - val_sdr_prev}\n')

        return val_sisnr

if __name__ == "__main__":
    args = parse_args()
    train_net(args)