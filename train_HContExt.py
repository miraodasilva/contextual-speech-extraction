import argparse
import random
import torch
from torch import nn, optim
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.distributed as dist
import time
import glob
import wandb
from datetime import datetime
import soundfile as sf
import shutil
from torch.utils.tensorboard import SummaryWriter
import contextlib, re
import torchmetrics
from torch.nn.attention import SDPBackend, sdpa_kernel
import idr_torch 

# model
from src.data.dataset_train_CSE import CSEDataset
from src.models.ContExt import Sepformer as HContExt
from transformers import LlamaModel
from speechbrain.inference.speaker import EncoderClassifier
from src.lr_scheduler import LinearWarmup, CosineAnnealingLRWarmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    parser.add_argument("--ctx_length", type=int, default=1, help='How long LLM ctx features will be employed')
    
    parser.add_argument("--num_max_mix", type=int, default=2, help='how many mixed speech will be used during training')
    parser.add_argument("--num_test_mix", type=int, default=2, help='Which evaluation set will be used 2 or 3 speaker mixed')
    parser.add_argument("--augmentation", default=False, action='store_true')
    parser.add_argument("--speed_perturb_ratio", type=str, default='0.9 1.0 1.1')
    parser.add_argument("--shift_prob", type=float, default=0.4)
    parser.add_argument("--max_shift_sec", type=float, default=0.5)
    parser.add_argument("--max_context_train", type=int, default=100, help='What maximum context will be used for training')

    parser.add_argument("--noise_add", default=False, action='store_true')

    parser.add_argument("--train_data", type=str, default='spokenwoz', help='dailytalk or spokenwoz or tedlium')
    parser.add_argument("--from_ckpt", default=False, action='store_true')

    parser.add_argument("--temp_dir", type=str, default='')
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/Sepformer')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--project", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--update_frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--tot_iters", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup", default=False, action='store_true')
    parser.add_argument("--warmup_iteration", type=int, default=10000)
    parser.add_argument("--plateau", default=False, action='store_true')
    parser.add_argument("--no_reduce", type=int, default=100000)
    parser.add_argument("--weight_decay", type=float, default=0.000001)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=5000)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--mode", type=str, default='train', help='train, test, valid')
    parser.add_argument("--reset_optimizer", default=False, action='store_true')

    parser.add_argument("--fp16", default=False, action='store_true')
    parser.add_argument("--bf16", default=False, action='store_true')

    parser.add_argument("--generate_speech", default=False, action='store_true', help='Whether saving audio during eval')
    parser.add_argument("--generate_step", type=int, default=1000)
    parser.add_argument("--num_gen_speech", type=int, default=20)

    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--torchrun", default=False, action='store_true')
    parser.add_argument("--masterport", type=str, default='1234')
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.rank)
    torch.cuda.manual_seed_all(args.rank)
    random.seed(args.rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = args.masterport

    if args.distributed:
        if args.torchrun:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
            )
            args.local_rank = int(os.environ['LOCAL_RANK'])
            args.rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(args.local_rank)
        else:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=idr_torch.size,
                rank=idr_torch.rank
            )
            args.local_rank = int(idr_torch.local_rank)
            args.rank = idr_torch.rank
            torch.cuda.set_device(args.local_rank)

    args.temp_dir = './tmp_eval/' + args.checkpoint_dir.split('/')[-1]
    if not os.path.exists(args.checkpoint_dir) and args.rank == 0:
        os.makedirs(args.checkpoint_dir)
    
    args.speed_perturb_ratio = [float(ratio) for ratio in args.speed_perturb_ratio.split()]

    train_data = CSEDataset(
        dailytalk_data_path=args.dailytalk_data_path,
        spokenwoz_data_path=args.spokenwoz_data_path, 
        tedlium_data_path=args.tedlium_data_path,
        llama_path=args.llama_path, 
        mode='train',
        train_data=args.train_data, 
        max_sp_len=args.max_sp_len,
        context_length=args.context_length,   #0 for all range
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
        noise_add=args.noise_add,
        return_16k_gt=True,
    )

    llm = LlamaModel.from_pretrained(args.llama_path, token=args.llama_auth_token, torch_dtype=torch.float16)
    llm.requires_grad_(False)
    llm.eval()

    spk_emb_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"},
    )

    spk_emb_model.requires_grad_(False)
    spk_emb_model.eval()

    model = HContExt(add_ctx=True, add_se=True, num_spks=args.num_max_mix)
    if args.from_ckpt:
        model.add_ctx_pipeline()
        model.add_se_pipeline()

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.resume and args.checkpoint is None:
        ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.ckpt')), key=lambda x: int(os.path.basename(x).split('_')[2]))
        if len(ckpts) != 0:
            args.checkpoint = ckpts[-1]
            if args.rank == 0:
                print(f'Resume with the latest checkpoint {ckpts[-1]}')
        else:
            if args.rank == 0:
                print(f'No checkpoint exists, start from scratch')

    if args.checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        if args.from_ckpt:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            scheduler_state_dict = checkpoint['scheduler_state_dict']
            args.start_step = checkpoint['step']
            args.start_epoch = checkpoint['epoch']
        else:
            optimizer_state_dict = None
            scheduler_state_dict = None
        del checkpoint
    else:
        optimizer_state_dict = None
        scheduler_state_dict = None

    if not args.from_ckpt:
        model.add_ctx_pipeline()
        model.add_se_pipeline()

    num_llm = sum(p.numel() for p in llm.parameters())
    num_model = sum(p.numel() for p in model.parameters())

    params = [
        {"params": [p for p in model.parameters()]},
    ]
    
    num_train = []
    for param in params:
        for p in param['params']:
            num_train.append(p.numel())

    num_train = sum(num_train)
    if args.rank == 0:
        print(f'Train # of params: {num_train:,} / {num_model:,}')
        print(f'LLM # of params: {num_llm:,}')

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    if optimizer_state_dict is not None:
        if args.rank == 0:
            print('* Optimizer is loaded from ckpt')
        optimizer.load_state_dict(optimizer_state_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if args.plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.0001)
    elif args.warmup:
        if args.tot_iters is not None:
            scheduler = CosineAnnealingLRWarmup(optimizer, T_max=args.tot_iters, T_warmup=args.warmup_iteration)
        else:
            scheduler = LinearWarmup(optimizer, T_warmup=args.warmup_iteration)
    else:
        scheduler = None

    if scheduler_state_dict is not None and scheduler is not None:
        if args.rank == 0:
            print('* Scheduler is loaded from ckpt')
        scheduler.load_state_dict(scheduler_state_dict)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    llm.cuda()
    model.cuda()
    if args.rank == 0:
        num_in_optimizer = []
        for param in optimizer.param_groups:
            for p in param['params']:
                num_in_optimizer.append(p.numel())
        print(f"Params in optimizer: {sum(num_in_optimizer)}")  # Make sure we're training

    if args.distributed:
        model = DDP(model, 
                    device_ids=[args.local_rank], 
                    output_device=args.local_rank, 
                    find_unused_parameters=True,
                    )

    _ = validate(model, llm, spk_emb_model, fast_validate=True, val_dataset=args.train_data)    # debug the pipeline
    train(model, llm, spk_emb_model, train_data, args.epochs, optimizer=optimizer, scheduler=scheduler, args=args, scaler=scaler)

def train(model, llm, spk_emb_model, train_data, epochs, optimizer, scheduler, args, scaler):
    best_val_sisnr = 0.
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")
    if args.rank == 0:
        if args.project is not None:
            wandbrun = wandb.init(
                project="CSE", 
                name=args.project + f'_{dt_string}', 
                id=get_id_from_experiment_dir(args.checkpoint_dir), 
                dir=args.checkpoint_dir
                )
            wandbrun.config.epochs = args.epochs
            wandbrun.config.batch_size = args.batch_size
            wandbrun.config.learning_rate = args.lr
            wandbrun.config.eval_step = args.eval_step
            wandbrun.config.update_frequency = args.update_frequency
            wandbrun.config.warmup = args.warmup
            wandbrun.config.warmup_iteration = args.warmup_iteration
            wandbrun.config.fp16 = args.fp16
            wandbrun.config.tot_iters = args.tot_iters
            wandbrun.config.context_length = args.context_length
            wandbrun.config.generate_speech = args.generate_speech
            wandbrun.config.max_sp_len = args.max_sp_len
            wandbrun.config.llama_path = args.llama_path
            wandbrun.config.num_max_mix = args.num_max_mix
            wandbrun.config.num_test_mix = args.num_test_mix
            wandbrun.config.augmentation = args.augmentation
            wandbrun.config.acoustic_noise_path = args.acoustic_noise_path
            wandbrun.config.speed_perturb_ratio = args.speed_perturb_ratio
            wandbrun.config.max_shift_sec = args.max_shift_sec
            wandbrun.config.shift_prob = args.shift_prob
            wandbrun.config.max_context_train = args.max_context_train
            wandbrun.config.train_data = args.train_data
            wandbrun.config.sr = args.sr
            writer = None
        else:
            writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])
            wandbrun = None
    else:
        writer = None
        wandbrun = None

    model.train()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, 
                                                                        num_replicas=None if args.torchrun else idr_torch.size, 
                                                                        rank=None if args.torchrun else idr_torch.rank,)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_data,
        shuffle=False if args.distributed else True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=lambda x: train_data.collate_fn(x),
    )

    criterion = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().cuda()

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = args.start_step

    optimizer.zero_grad()
    for epoch in range(args.start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.rank == 0:
            print(f"Epoch [{epoch}/{epochs}]")
            prev_time = time.time()
        for i, batch in enumerate(dataloader):
            if args.rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %f ********" % (
                    epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['lr']))
            mixed_sp, gt_sp, context, context_mask, sp_len, _, gt_16k, sp_16k_len = batch

            llm_output = llm(input_ids=context.cuda(), attention_mask=context_mask.cuda())
            ctx_feat = llm_output.last_hidden_state[:, -args.ctx_length:]
            if not args.fp16:
                ctx_feat = ctx_feat.float()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if args.fp16 or args.bf16 else contextlib.nullcontext(), torch.autocast(device_type='cuda', dtype=torch.float16 if args.fp16 else torch.bfloat16) if args.fp16 or args.bf16 else contextlib.nullcontext():
                spk_embed = spk_emb_model.encode_batch(gt_16k.cuda(), wav_lens=(sp_16k_len / max(sp_16k_len)))
                enhanced_sp = model(mixed_sp.cuda(), ctx_feat, spk_embed)[:, :, 0]
                loss = -1. * criterion(enhanced_sp, gt_sp.cuda())
                snr_loss = loss.clone()

            if args.fp16:
                scaler.scale(loss).backward()
                loss = loss.float()
            else:
                loss.backward()

            if ((i + 1) % args.update_frequency == 0) or (i + 1 == len(dataloader)):
                step += 1
                if args.fp16:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    if not torch.isfinite(grad_norm) and args.rank == 0:
                        print(f"The grad norm is {grad_norm}. Skipping updating the model.")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    if not torch.isfinite(grad_norm) and args.rank == 0:
                        print(f"The grad norm is {grad_norm}. Skipping updating the model.")
                    else:
                        optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and not args.plateau:
                    scheduler.step()
            else:
                continue

            if args.rank == 0:
                if writer is not None:
                    writer.add_scalar('train/loss', loss.cpu().item(), step)
                    writer.add_scalar('train/SI-SNR', -1. * snr_loss.cpu().item(), step)
                    writer.add_scalar('lr/learning_rate', optimizer.param_groups[0]['lr'], step)
 
                if step % 100 == 0:
                    if args.fp16:
                        print(f'######## Step(Epoch): {step}({epoch}), Loss: {loss.cpu().item()}, Scale: {scaler.get_scale()} #########')
                    else:
                        print(f'######## Step(Epoch): {step}({epoch}), Loss: {loss.cpu().item()} #########')
                        
                    if wandbrun is not None:
                        wandbrun.log({'train/loss': loss.cpu().item()}, step)
                        wandbrun.log({'train/SI-SNR': -1. * snr_loss.cpu().item()}, step)
                        wandbrun.log({'train/learning_rate': optimizer.param_groups[0]['lr']}, step)
            
            if step % args.eval_step == 0:
                val_sisnr = validate(model, llm, spk_emb_model, epoch=epoch, writer=writer, wandbrun=wandbrun, step=step, val_dataset=args.train_data)
                model.train()

                if args.plateau and step >= args.no_reduce:
                    scheduler.step(val_sisnr)

                if args.distributed:
                    dist.barrier()

                if args.rank == 0:
                    print('VAL_SI-SNR: ', val_sisnr)
                    print('Saving checkpoint for Epoch: %d' % epoch)
                    if args.distributed:
                        state_dict = model.module.state_dict()
                        optimizer_state_dict = optimizer.state_dict()
                        if scheduler is not None:
                            scheduler_state_dict = scheduler.state_dict()
                    else:
                        state_dict = model.state_dict()
                        optimizer_state_dict = optimizer.state_dict()
                        if scheduler is not None:
                            scheduler_state_dict = scheduler.state_dict()
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save(
                        {
                        'state_dict': state_dict,
                        'optimizer_state_dict': optimizer_state_dict,
                        'scheduler_state_dict': scheduler_state_dict if scheduler is not None else None,
                        'step': step,
                        'epoch': epoch,
                        },
                        os.path.join(args.checkpoint_dir, 'Epoch_%04d_%05d_%.2f.ckpt' % (epoch, step, val_sisnr)))

                    if val_sisnr >= best_val_sisnr:
                        best_val_sisnr = val_sisnr
                        bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                        for prev in bests:
                            os.remove(prev)
                        torch.save(
                            {
                            'state_dict': state_dict,
                            'optimizer_state_dict': optimizer_state_dict,
                            'scheduler_state_dict': scheduler_state_dict if scheduler is not None else None,
                            'step': step,
                            'epoch': epoch,
                             },
                            os.path.join(args.checkpoint_dir, 'Best_%04d_%05d_%.2f.ckpt' % (epoch, step, val_sisnr)))

            if args.generate_speech and step % args.generate_step == 0:
                if args.rank == 0:
                    if os.path.exists(os.path.join(args.temp_dir, 'train')):
                        shutil.rmtree(os.path.join(args.temp_dir, 'train'))
                    for kk, (gt_sp, pred_sp, mixed, length) in enumerate(zip(gt_sp, enhanced_sp.cpu().detach(), mixed_sp, sp_len)):
                        gt_save_name = os.path.join(args.temp_dir, 'train', 'gts', f'{kk}.wav')
                        pred_save_name = os.path.join(args.temp_dir, 'train', 'preds', f'{kk}.wav')
                        mixed_save_name = os.path.join(args.temp_dir, 'train', 'mixed', f'{kk}.wav')
                        if not os.path.exists(os.path.dirname(gt_save_name)):
                            os.makedirs(os.path.dirname(gt_save_name))
                        if not os.path.exists(os.path.dirname(pred_save_name)):
                            os.makedirs(os.path.dirname(pred_save_name))
                        if not os.path.exists(os.path.dirname(mixed_save_name)):
                            os.makedirs(os.path.dirname(mixed_save_name)) 
                        
                        gt_sp = gt_sp.float().numpy()
                        gt_sp = gt_sp / np.max(np.abs(gt_sp)) * 0.9
                        pred_sp = pred_sp.float().numpy()
                        pred_sp = pred_sp / np.max(np.abs(pred_sp)) * 0.9
                        mixed = mixed.float().numpy()
                        mixed = mixed / np.max(np.abs(mixed)) * 0.9

                        sf.write(gt_save_name, gt_sp, samplerate=args.sr, subtype='PCM_16')
                        sf.write(pred_save_name, pred_sp, samplerate=args.sr, subtype='PCM_16')
                        sf.write(mixed_save_name, mixed, samplerate=args.sr, subtype='PCM_16')  
                        if wandbrun is not None:
                            if kk < 3:
                                wandbrun.log({f'train/GT/{kk}': wandb.Audio(gt_sp, sample_rate=args.sr)}, step)
                                wandbrun.log({f'train/Pred/{kk}': wandb.Audio(pred_sp, sample_rate=args.sr)}, step)
                                wandbrun.log({f'train/Mixed/{kk}': wandb.Audio(mixed, sample_rate=args.sr)}, step)

                if args.distributed:
                    dist.barrier()

            if (step - 1) == args.tot_iters:
                if args.distributed:
                    dist.barrier()
                assert 1 == 0, 'Total Iteration Reached'

    if args.rank == 0:
        print('Finishing training')

def validate(model, llm, spk_emb_model, fast_validate=False, epoch=0, writer=None, wandbrun=None, step=0, val_dataset='spokenwoz'):
    assert val_dataset in ['spokenwoz', 'dailytalk', 'msp-conversation', 'tedlium']
    with torch.no_grad():
        model.eval()

        val_data = CSEDataset(
            dailytalk_data_path=args.dailytalk_data_path,
            spokenwoz_data_path=args.spokenwoz_data_path, 
            tedlium_data_path=args.tedlium_data_path,
            llama_path=args.llama_path,
            mode='val',
            train_data=val_dataset, 
            max_sp_len=30,  
            context_length=args.context_length,   #0 for all range
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
            noise_add=args.noise_add,
            return_noise=True,
            return_16k_gt=True,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            collate_fn=lambda x: val_data.collate_fn(x),
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(5 * batch_size, int(len(dataloader.dataset)))
            max_batches = 5
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        criterion = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().cuda()

        val_sisnr_list = []
        prev_sisnr_list = []
        gts = []
        preds = []
        mixed = []
        sp_lens = []
        f_names = []

        if args.rank == 0:
            if os.path.exists(os.path.join(args.temp_dir, 'val', 'preds')) \
            or os.path.exists(os.path.join(args.temp_dir, 'val', 'gts')) \
            or os.path.exists(os.path.join(args.temp_dir, 'val', 'mixed')):
                shutil.rmtree(args.temp_dir)

        description = 'Validation on subset of the Val dataset' if fast_validate else 'Validation'
        if args.rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.rank == 0 and i % 100 == 0:
                if not fast_validate:
                    print(f"******** Validation ({val_dataset}) : {(i + 1) * batch_size} / {samples} ********")
            if args.num_max_mix == 2:
                mixed_sp, gt_sp, context, context_mask, sp_len, f_name, gt_16k, sp_16k_len, ns_sp_1 = batch
            elif args.num_max_mix == 3:
                mixed_sp, gt_sp, context, context_mask, sp_len, f_name, gt_16k, sp_16k_len, ns_sp_1, ns_sp_2 = batch

            llm_output = llm(input_ids=context.cuda(), attention_mask=context_mask.cuda())
            ctx_feat = llm_output.last_hidden_state[:, -args.ctx_length:]
            if not args.fp16:
                ctx_feat = ctx_feat.float()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if args.fp16 or args.bf16 else contextlib.nullcontext(), torch.autocast(device_type='cuda', dtype=torch.float16 if args.fp16 else torch.bfloat16) if args.fp16 or args.bf16 else contextlib.nullcontext():
                spk_embed = spk_emb_model.encode_batch(gt_16k.cuda(), wav_lens=(sp_16k_len / max(sp_16k_len)))
                enhanced_sp = model(mixed_sp.cuda(), ctx_feat, spk_embed)[:, :, 0]
                loss = -1. * criterion(enhanced_sp, gt_sp.cuda())
                prev = criterion(mixed_sp.cuda(), gt_sp.cuda())

            loss = loss.float().cpu().item()

            val_sisnr_list.append(loss * -1.)
            prev_sisnr_list.append(prev.float().cpu().item())

            if args.generate_speech and len(mixed) < args.num_gen_speech:
                mixed.extend([sp for sp in mixed_sp])
                preds.extend([sp for sp in enhanced_sp.cpu()])
                gts.extend([sp for sp in gt_sp])
                sp_lens.extend([sp for sp in sp_len])
                f_names.extend(f_name)

            if args.distributed:
                dist.barrier()

            if i >= max_batches:
                break

        if args.generate_speech:
            if args.rank == 0:
                if os.path.exists(os.path.join(args.temp_dir, 'val')):
                    shutil.rmtree(os.path.join(args.temp_dir, 'val'))
                for kk, (gt_sp, pred_sp, mixed_sp, sp_len, f_name) in enumerate(zip(gts, preds, mixed, sp_lens, f_names)):
                    gt_save_name = os.path.join(args.temp_dir, 'val', val_dataset, 'gts', f_name + '.wav')
                    pred_save_name = os.path.join(args.temp_dir, 'val', val_dataset, 'preds', f_name + '.wav')
                    mixed_save_name = os.path.join(args.temp_dir, 'val', val_dataset, 'mixed', f_name + '.wav')
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
                    if wandbrun is not None:
                        if kk < 3:
                            wandbrun.log({f'val_{val_dataset}/GT/{kk}': wandb.Audio(gt_sp, sample_rate=args.sr)}, step)
                            wandbrun.log({f'val_{val_dataset}/Pred/{kk}': wandb.Audio(pred_sp, sample_rate=args.sr)}, step)
                            wandbrun.log({f'val_{val_dataset}/Mixed/{kk}': wandb.Audio(mixed_sp, sample_rate=args.sr)}, step)

        if args.distributed:
            dist.barrier()

        val_sisnr = np.mean(val_sisnr_list)
        prev_sisnr = np.mean(prev_sisnr_list)
        if args.rank == 0:
            print(f"## VALIDATION SI-SNR ({val_dataset}): ", val_sisnr)      

        if args.rank == 0:
            if writer is not None:
                writer.add_scalar(f'val_{val_dataset}/SI-SNR', val_sisnr, step)
                writer.add_scalar(f'val_{val_dataset}/SI-SNR-I', val_sisnr - prev_sisnr, step)
            if wandbrun is not None:
                wandbrun.log({f'val_{val_dataset}/SI-SNR': val_sisnr}, step)
                wandbrun.log({f'val_{val_dataset}/SI-SNR-I': val_sisnr - prev_sisnr}, step)
        return val_sisnr

def get_id_from_experiment_dir(experiment_dir = './'):
    wandb_files = glob.glob(os.path.join(experiment_dir, "wandb/latest-run/run-*.wandb"))
    if len(wandb_files) != 0:
        wandb_file = wandb_files[0]
        return re.search("run-(.+?).wandb", wandb_file).group(1)
    else:
        return None

if __name__ == "__main__":
    args = parse_args()
    train_net(args)