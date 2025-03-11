import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import re
import librosa
import glob
import random
from transformers import AutoTokenizer
import torchaudio.functional as F

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def text_process(text):
    text = text.replace("[unk]", "")
    text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
    return text

class CSEDataset(Dataset):
    def __init__(
            self,
            dailytalk_data_path='./DailyTalk_processed_16k',
            spokenwoz_data_path='./Spokenwoz_preprocessed',
            tedlium_data_path='./TEDLIUM_release-3_CSF',
            train_data='dailytalk',
            llama_path='meta-llama/Meta-Llama-3-8B', #'meta-llama/Llama-2-7b-hf',
            mode='train', 
            max_sp_len=16,  #16 sec
            context_length=0,   #0 for using full context (::Inference)
            auth_token=None,
            num_max_mix=2,
            num_test_mix=2,
            augmentation=True,
            acoustic_noise_path='./DEMAND',    # original data contains 16k and 32k samplerate samples
            speed_perturb_ratio=[0.9, 1.0, 1.1],
            max_shift_sec=0.5,
            max_context_train=300,  #(::Train)
            sr=8000,
            shift_prob=0.4,
            return_16k_gt=False,
            noise_add=True,
            return_noise=False,
            one_sec=False,
            ):
        assert mode in ['train', 'test', 'val']
        assert train_data in ['dailytalk', 'spokenwoz', 'tedlium']

        self.mode = mode
        self.num_max_mix = num_max_mix
        assert num_max_mix == num_test_mix
        self.speed_perturb_ratio = speed_perturb_ratio
        self.max_shift_sec = max_shift_sec
        self.max_context_train = max_context_train
        self.sr = sr
        self.shift_prob = shift_prob
        self.tedlium_data_path = tedlium_data_path
        self.context_length = context_length
        self.return_16k_gt = return_16k_gt
        self.one_sec = one_sec

        self.train_data = train_data
        self.noise_add = noise_add
        self.return_noise = return_noise

        if mode == 'test':
            if train_data == 'dailytalk':
                self.test_limit = 5
                print(f"* * * Use samples have more than 5 context window")
            else:
                self.test_limit = 10
                print(f"* * * Use samples have more than {self.test_limit} context window")

        if train_data == 'dailytalk':
            data_path = dailytalk_data_path
        elif train_data == 'spokenwoz':
            data_path = spokenwoz_data_path
        elif train_data == 'tedlium':
            data_path = tedlium_data_path

        self.acoustic_noises = sorted(glob.glob(os.path.join(acoustic_noise_path, '*', '*.wav'))) 

        if augmentation and mode == 'train':
            self.augmentation = True
        else:
            self.augmentation = False

        if mode == 'train':
            self.f_paths = self.build_train_list(
                data_path, 
                mode,
                train_data, 
            )
            self.gt_paths = None
        else:
            print(f"* * * Evaluation will be peformed on {num_test_mix} speaker mixed samples of {train_data}")
            self.f_paths, self.gt_paths = self.build_eval_list(
                data_path,
                mode,
                num_test_mix,
            )
            if mode == 'test':
                print(f"Num test files: {len(self.f_paths)}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, token=auth_token)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.max_sp_len = max_sp_len * 16000

        # For the case when the speaker embedding is used.
        self.dailytalk_register = {
            '0': os.path.join(dailytalk_data_path, 'test/gt/237_0_0_d237-72_4_1_d72-3.9282.wav'),
            '1': os.path.join(dailytalk_data_path, 'test/gt/32_0_1_d32-1405_0_0_d1405-3.9264.wav')
        }

    def build_train_list(self, path, mode, train_data):
        assert mode == 'train'
        if train_data != 'tedlium':
            file_paths = []
            if train_data == 'dailytalk':
                with open('./data/DailyTalk/train_dialog.txt', 'r') as txt:
                    lines = txt.readlines()
                for l in lines:
                    train_dialog = os.path.join(path, mode, l.strip())
                    files = sorted(glob.glob(os.path.join(train_dialog, "*.wav")))
                    file_paths.extend(files)
            elif train_data == 'spokenwoz':
                dialogs = sorted(os.listdir(os.path.join(path, mode)))
                for dialog in dialogs:
                    train_dialog = os.path.join(path, mode, dialog)
                    files = sorted(glob.glob(os.path.join(train_dialog, "*.wav")))
                    file_paths.extend(files)
        else:
            file_paths = sorted(glob.glob(os.path.join(path, mode, '*', '*.wav')))
        return file_paths

    def build_eval_list(self, path, mode, num_test_mix):
        assert mode in ['val', 'test']
        file_paths, gt_paths = [], []

        mix_name = 'mixed' if num_test_mix == 2 else 'mixed_3speaker'
        gt_name = 'gt' if num_test_mix == 2 else 'gt_3speaker'

        files = sorted(glob.glob(os.path.join(path, mode, mix_name, '*.txt')))
        for f in files:
            if mode == 'test':
                with open(f, 'r') as txt:
                    lines = txt.readlines()
                if len(lines) < self.test_limit:
                    continue
            file_paths.append(f[:-4] + '.wav')
            fpaths = os.path.normpath(f).split(os.sep)
            fpaths[-2] = gt_name
            gt_paths.append(os.sep.join(fpaths)[:-4] + '.wav')
        
        if self.train_data == 'spokenwoz' and mode == 'val': # val set is too large, only use 1000 for speed up during training
            index = np.random.permutation(np.arange(len(file_paths), dtype=int))[:1000]
            file_paths = list(np.array(file_paths)[index])
            gt_paths = list(np.array(gt_paths)[index])
        return file_paths, gt_paths

    def __len__(self):
        return len(self.f_paths)

    def __getitem__(self, idx):
        f_path = self.f_paths[idx]

        if self.mode == 'train':
            if self.num_max_mix == 2:
                noise_file = random.sample(list(set(self.f_paths) - set([f_path])), 1)[0]
                noise_aud, _ = librosa.load(noise_file, sr=16000)
                noise_aud = noise_aud / np.max(np.abs(noise_aud)) * 0.9
                if self.augmentation:
                    noise_aud = torch.tensor(noise_aud)

                    # Random Shifting
                    if random.random() < self.shift_prob:
                        shift_noise = random.randint(-1. * self.max_shift_sec * 16000, self.max_shift_sec * 16000)
                        noise_aud = torch.roll(noise_aud, shifts=shift_noise, dims=0)

                    # Speed Pertubation
                    ratio_noise = random.randint(0, len(self.speed_perturb_ratio) - 1)
                    noise_aud, _ = F.speed(noise_aud, orig_freq=16000, factor=self.speed_perturb_ratio[ratio_noise])

                    noise_aud = noise_aud.numpy()

                if len(noise_aud) > self.max_sp_len:
                    noise_aud = noise_aud[:self.max_sp_len]
                noise_aud_1 = noise_aud.copy()

            elif self.num_max_mix == 3:
                noise_file = random.sample(list(set(self.f_paths) - set([f_path])), 2)

                noise_aud_1, _ = librosa.load(noise_file[0], sr=16000)
                noise_aud_1 = noise_aud_1 / np.max(np.abs(noise_aud_1)) * 0.9
                if self.augmentation:
                    noise_aud_1 = torch.tensor(noise_aud_1)

                    # Random Shifting
                    if random.random() < self.shift_prob:
                        shift_noise = random.randint(-1. * self.max_shift_sec * 16000, self.max_shift_sec * 16000)
                        noise_aud_1 = torch.roll(noise_aud_1, shifts=shift_noise, dims=0)

                    # Speed Pertubation
                    ratio_noise = random.randint(0, len(self.speed_perturb_ratio) - 1)
                    noise_aud_1, _ = F.speed(noise_aud_1, orig_freq=16000, factor=self.speed_perturb_ratio[ratio_noise])

                    noise_aud_1 = noise_aud_1.numpy()
                
                if len(noise_aud_1) > self.max_sp_len:
                    noise_aud_1 = noise_aud_1[:self.max_sp_len]
                noise_aud_1 = noise_aud_1.copy()

                noise_aud_2, _ = librosa.load(noise_file[1], sr=16000)
                noise_aud_2 = noise_aud_2 / np.max(np.abs(noise_aud_2)) * 0.9
                if self.augmentation:
                    noise_aud_2 = torch.tensor(noise_aud_2)

                    # Random Shifting
                    if random.random() < self.shift_prob:
                        shift_noise = random.randint(-1. * self.max_shift_sec * 16000, self.max_shift_sec * 16000)
                        noise_aud_2 = torch.roll(noise_aud_2, shifts=shift_noise, dims=0)

                    # Speed Pertubation
                    ratio_noise = random.randint(0, len(self.speed_perturb_ratio) - 1)
                    noise_aud_2, _ = F.speed(noise_aud_2, orig_freq=16000, factor=self.speed_perturb_ratio[ratio_noise])

                    noise_aud_2 = noise_aud_2.numpy()
                
                if len(noise_aud_2) > self.max_sp_len:
                    noise_aud_2 = noise_aud_2[:self.max_sp_len]
                noise_aud_2 = noise_aud_2.copy()

            source_aud, _ = librosa.load(f_path, sr=16000)
            source_aud = source_aud / np.max(np.abs(source_aud)) * 0.9
            if self.augmentation:
                source_aud = torch.tensor(source_aud)
                
                # Random Shifting
                if random.random() < self.shift_prob:
                    shift_source = random.randint(-1. * self.max_shift_sec * 16000, self.max_shift_sec * 16000)
                    source_aud = torch.roll(source_aud, shifts=shift_source, dims=0)

                # Speed Pertubation
                ratio_source = random.randint(0, len(self.speed_perturb_ratio) - 1)
                source_aud, _ = F.speed(source_aud, orig_freq=16000, factor=self.speed_perturb_ratio[ratio_source])

                source_aud = source_aud.numpy()

            if len(source_aud) > self.max_sp_len:
                source_aud = source_aud[:self.max_sp_len]
            gt_16k_aud = source_aud.copy()

            if self.num_max_mix == 2:
                snr = np.clip(random.normalvariate(0, 4), -5, 5)
                if random.random() < 0.5:
                    # with half prob. Use full source_aud length and noise_aud will be cut or pad
                    mixed_aud, source_aud, noise_aud = self.mix_audio(source_aud, noise_aud, snr, pad=True)
                else:
                    # with half prob. Use full noise_aud length and source_aud will be cut or pad
                    mixed_aud, noise_aud, source_aud = self.mix_audio(noise_aud, source_aud, snr, pad=True)
            elif self.num_max_mix == 3:
                snr1, snr2 = np.clip(random.normalvariate(0, 4), -5, 5), np.clip(random.normalvariate(0, 4), -5, 5)
                mixed_aud, source_aud, noise_aud_1, noise_aud_2 = self.mix_audio_3spk(source_aud, noise_aud_1, noise_aud_2, snr1, snr2, pad=True)

            if len(mixed_aud) > len(source_aud):
                source_aud = np.concatenate([source_aud, np.zeros(len(mixed_aud) - len(source_aud))], 0)
            if len(mixed_aud) < len(source_aud):
                source_aud = source_aud[:len(mixed_aud)]
            if len(mixed_aud) > len(noise_aud_1):
                noise_aud_1 = np.concatenate([noise_aud_1, np.zeros(len(mixed_aud) - len(noise_aud_1))], 0)
            if len(mixed_aud) < len(noise_aud_1):
                noise_aud_1 = noise_aud_1[:len(mixed_aud)]
            if self.num_max_mix == 3:
                if len(mixed_aud) > len(noise_aud_2):
                    noise_aud_2 = np.concatenate([noise_aud_2, np.zeros(len(mixed_aud) - len(noise_aud_2))], 0)
                if len(mixed_aud) < len(noise_aud_2):
                    noise_aud_2 = noise_aud_2[:len(mixed_aud)]

            gt_aud = source_aud.copy()

            if self.augmentation and self.noise_add:
                # Acoustic Noise Addition
                if random.random() < 0.5:   # Acoustic noise is added with 50 percent chance
                    acoustic_noise_file = random.choice(self.acoustic_noises)
                    acoustic_noise, _ = librosa.load(acoustic_noise_file, sr=16000)
                    acoustic_noise = acoustic_noise / np.max(np.abs(acoustic_noise)) * 0.9
                    
                    select_length = len(mixed_aud)
                    if select_length > len(acoustic_noise):
                        acoustic_noise = acoustic_noise[np.arange(select_length) % len(acoustic_noise)]
                    start_ind = random.randint(0, len(acoustic_noise) - select_length)
                    acoustic_noise = acoustic_noise[start_ind:start_ind + select_length]
                    
                    ac_noise_snr = random.random() * 10
                    mixed_aud = F.add_noise(torch.tensor(mixed_aud), torch.tensor(acoustic_noise), torch.tensor(ac_noise_snr)).numpy()

            context = []            
            # context load
            with open(os.path.splitext(f_path)[0] + '.txt', 'r') as txt:
                lines = txt.readlines()
            if len(lines) > 0:
                if self.tedlium_data_path in f_path:
                    for spk, line in enumerate(lines):
                        context += [text_process(line.strip())]
                else:
                    for spk, line in enumerate(lines):
                        context += [f'Speaker {spk % 2}: ' + text_process(line.strip())]
                
                # train with random context size
                context_window = random.randint(1, min(len(context), self.max_context_train))
                context = context[-context_window:]
            else:
                spk = 0        

            if self.tedlium_data_path in f_path:
                context += ['']
            else:
                context += [f'Speaker {(spk + 1)%2}: ']
            context = '/n'.join(context)
        else:
            gt_path = self.gt_paths[idx]
            mixed_aud, _ = librosa.load(f_path, sr=16000)
            gt_aud, _ = librosa.load(gt_path, sr=16000)
            
            if self.num_max_mix == 2:
                noise_aud_1, _ = librosa.load(gt_path.replace('gt', 'noise'), sr=16000)
            elif self.num_max_mix == 3:
                noise_aud_1, _ = librosa.load(gt_path.replace('gt', 'noise_1'), sr=16000)
                noise_aud_2, _ = librosa.load(gt_path.replace('gt', 'noise_2'), sr=16000)

            if len(mixed_aud) > self.max_sp_len:
                print(f'Speech is truncated due to exceeding length: {len(mixed_aud)}')
                mixed_aud = mixed_aud[:self.max_sp_len]
                gt_aud = gt_aud[:self.max_sp_len]

            context = []
            # context load
            with open(os.path.splitext(f_path)[0] + '.txt', 'r') as txt:
                lines = txt.readlines()
            if len(lines) > 0:
                if self.tedlium_data_path in f_path:
                    for spk, line in enumerate(lines):
                        context += [text_process(line.strip())]
                else:
                    for spk, line in enumerate(lines):
                        context += [f'Speaker {spk % 2}: ' + text_process(line.strip())]

                if self.context_length > 0:
                    context = context[-self.context_length:]
                elif self.context_length == -1:
                    context = []
            else:
                spk = 0

            if self.tedlium_data_path in f_path:
                context += [f'']
            else:
                context += [f'Speaker {(spk + 1)%2}: ']

            context = '/n'.join(context)
            
            if len(gt_aud) > len(noise_aud_1):
                noise_aud_1 = np.concatenate([noise_aud_1, np.zeros(len(gt_aud) - len(noise_aud_1))], 0)
            if len(gt_aud) < len(noise_aud_1):
                noise_aud_1 = noise_aud_1[:len(gt_aud)]
            if self.num_max_mix == 3:
                if len(gt_aud) > len(noise_aud_2):
                    noise_aud_2 = np.concatenate([noise_aud_2, np.zeros(len(gt_aud) - len(noise_aud_2))], 0)
                if len(gt_aud) < len(noise_aud_2):
                    noise_aud_2 = noise_aud_2[:len(gt_aud)]
        
        if self.return_16k_gt:
            if self.mode == 'train':
                embed_length = random.randint(1, 5)
                st_ind = random.randint(0, max(0, len(gt_16k_aud) - int(16000 * embed_length)))
                gt_16k_aud = gt_16k_aud[st_ind:st_ind + int(16000 * embed_length)]
            elif self.mode != 'train':
                if self.one_sec or self.train_data == 'spokenwoz':
                    st_ind = random.randint(0, max(0, len(gt_16k_aud) - int(16000 * 1)))
                    gt_16k_aud = gt_16k_aud[st_ind:st_ind + int(16000 * 1)]
                else:
                    if self.train_data == 'tedlium':
                        spk_name = os.path.basename(f_path).split('-')[0]
                        candidate = sorted(glob.glob(os.path.join(self.tedlium_data_path, self.mode, 'gt' if self.num_max_mix == 2 else 'gt_3speaker', f'{spk_name}*.wav')))[0]
                        gt_16k_aud, _ = librosa.load(candidate, sr=16000)
                    elif self.train_data == 'dailytalk':
                        spk_name = os.path.basename(f_path).split('_')[2]
                        gt_16k_aud, _ = librosa.load(self.dailytalk_register[spk_name], sr=16000)
                    
        if self.sr != 16000:
            mixed_aud = librosa.resample(mixed_aud, orig_sr=16000, target_sr=self.sr)
            gt_aud = librosa.resample(gt_aud, orig_sr=16000, target_sr=self.sr)
            noise_aud_1 = librosa.resample(noise_aud_1, orig_sr=16000, target_sr=self.sr)
            if self.num_max_mix > 2:
                noise_aud_2 = librosa.resample(noise_aud_2, orig_sr=16000, target_sr=self.sr)

        if self.return_16k_gt:
            if self.return_noise:
                if self.num_max_mix > 2:
                    return mixed_aud, gt_aud, context, os.path.splitext(os.path.basename(f_path))[0], gt_16k_aud, noise_aud_1, noise_aud_2
                else:
                    return mixed_aud, gt_aud, context, os.path.splitext(os.path.basename(f_path))[0], gt_16k_aud, noise_aud_1
            else:
                return mixed_aud, gt_aud, context, os.path.splitext(os.path.basename(f_path))[0], gt_16k_aud
        else:
            if self.return_noise:
                if self.num_max_mix > 2:
                    return mixed_aud, gt_aud, context, os.path.splitext(os.path.basename(f_path))[0], noise_aud_1, noise_aud_2
                else:
                    return mixed_aud, gt_aud, context, os.path.splitext(os.path.basename(f_path))[0], noise_aud_1
            else:
                return mixed_aud, gt_aud, context, os.path.splitext(os.path.basename(f_path))[0]

    def mix_audio(self, signal, noise, snr, pad=False):
        # if the audio is longer than the noise
        # if pad is true the noise is zero padded to match the length.
        # else play the noise in repeat for the duration of the audio
        if not pad and len(signal) > len(noise):
            noise = noise[np.arange(len(signal)) % len(noise)]
        if len(signal) < len(noise):
            noise = noise[:len(signal)]
        # this is important if loading resulted in 
        # uint8 or uint16 types, because it would cause overflow
        # when squaring and calculating mean
        noise = noise.astype(np.float32)
        signal = signal.astype(np.float32)
        
        # get the initial energy for reference
        signal_energy = np.mean(signal**2)
        noise_energy = np.mean(noise**2)
        # calculates the gain to be applied to the noise 
        # to achieve the given SNR
        g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
        
        # Assumes signal and noise to be decorrelated
        # and calculate (a, b) such that energy of 
        # a*signal + b*noise matches the energy of the input signal
        a = np.sqrt(1 / (1 + g**2))
        b = np.sqrt(g**2 / (1 + g**2))

        if pad and len(signal) > len(noise):
            noise = np.concatenate([noise, np.zeros(len(signal) - len(noise))], 0)
        # mix the signals
        signal = a * signal
        noise = b * noise

        mixed_audio = signal + noise

        scale = 1 / np.max(np.abs(mixed_audio)) * 0.9
        mixed_audio = scale * mixed_audio
        signal = scale * signal
        noise = scale * noise
        return mixed_audio, signal, noise

    def mix_audio_3spk(self, signal, noise1, noise2, snr1, snr2, pad=False):
        # if the audio is longer than the noise
        # if pad is true the noise is zero padded to match the length.
        # else play the noise in repeat for the duration of the audio
        max_len = max([len(signal), len(noise1), len(noise2)])
        if not pad:
            if max_len > len(signal):
                signal = signal[np.arange(max_len) % len(signal)]
            if max_len > len(noise1):
                noise1 = noise1[np.arange(max_len) % len(noise1)]
            if max_len > len(noise2):
                noise2 = noise2[np.arange(max_len) % len(noise2)]
            
        # this is important if loading resulted in 
        # uint8 or uint16 types, because it would cause overflow
        # when squaring and calculating mean
        noise1 = noise1.astype(np.float32)
        noise2 = noise2.astype(np.float32)
        signal = signal.astype(np.float32)
        
        # get the initial energy for reference
        signal_energy = np.mean(signal**2)
        noise1_energy = np.mean(noise1**2)
        noise2_energy = np.mean(noise2**2)

        # calculates the gain to be applied to the noise to achieve the given SNR
        g1 = np.sqrt(10.0 ** (-snr1/10) * signal_energy / noise1_energy)
        g2 = np.sqrt(10.0 ** (-snr2/10) * signal_energy / noise2_energy)

        if pad:
            if max_len > len(signal):
                signal = np.concatenate([signal, np.zeros(max_len - len(signal))], 0)
            if max_len > len(noise1):
                noise1 = np.concatenate([noise1, np.zeros(max_len - len(noise1))], 0)
            if max_len > len(noise2):
                noise2 = np.concatenate([noise2, np.zeros(max_len - len(noise2))], 0)

        noise1 = g1 * noise1
        noise2 = g2 * noise2

        mixed_audio = signal + noise1 + noise2

        scale = 1 / np.max(np.abs(mixed_audio)) * 0.9
        mixed_audio = scale * mixed_audio
        signal = scale * signal
        noise1 = scale * noise1
        noise2 = scale * noise2
        return mixed_audio, signal, noise1, noise2

    def collate_fn(self, batch):
        # mixed_aud, gt_aud, ctx_txt, os.path.splitext(os.path.basename(f_path))[0]
        sp_len, sp_16k_len, ctxs, f_names = [], [], [], []
        for data in batch:
            sp_len.append(len(data[0]))
            ctxs.append(data[2])
            f_names.append(data[3])
            if self.return_16k_gt:
                sp_16k_len.append(len(data[4]))

        max_sp_len = max(sp_len)
        if self.return_16k_gt:
            max_sp_16k_len = max(sp_16k_len)

        padded_mixed_sp = []
        padded_gt_sp = []
        padded_gt_16k = []
        padded_ns_sp_1 = []
        padded_ns_sp_2 = []

        if self.return_16k_gt:
            if self.return_noise:
                if self.num_max_mix > 2:
                    for mixed_sp, gt_sp, _, _, gt_16k, noise_sp_1, noise_sp_2 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_gt_16k.append(np.concatenate([gt_16k, np.zeros([max_sp_16k_len - len(gt_16k)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
                        padded_ns_sp_2.append(np.concatenate([noise_sp_2, np.zeros([max_sp_len - len(noise_sp_2)])], axis=0))
                else:
                    for mixed_sp, gt_sp, _, _, gt_16k, noise_sp_1 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_gt_16k.append(np.concatenate([gt_16k, np.zeros([max_sp_16k_len - len(gt_16k)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
            else:
                for mixed_sp, gt_sp, _, _, gt_16k in batch:
                    # right padding
                    padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                    padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                    padded_gt_16k.append(np.concatenate([gt_16k, np.zeros([max_sp_16k_len - len(gt_16k)])], axis=0))
        else:
            if self.return_noise:
                if self.num_max_mix > 2:
                    for mixed_sp, gt_sp, _, _, noise_sp_1, noise_sp_2 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
                        padded_ns_sp_2.append(np.concatenate([noise_sp_2, np.zeros([max_sp_len - len(noise_sp_2)])], axis=0))
                else:
                    for mixed_sp, gt_sp, _, _, noise_sp_1 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
            else:
                for mixed_sp, gt_sp, _, _, in batch:
                    # right padding
                    padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                    padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))

        # left padding
        ctx_txt = self.tokenizer(ctxs, return_tensors="pt", padding=True)

        mixed_sp_inp = torch.FloatTensor(np.array(padded_mixed_sp))
        gt_sp_inp = torch.FloatTensor(np.array(padded_gt_sp))
        ctx_inp = ctx_txt.input_ids
        ctx_pad = ctx_txt.attention_mask
        sp_len = torch.IntTensor(sp_len)
        if self.return_noise:
            noise_sp_inp_1 = torch.FloatTensor(np.array(padded_ns_sp_1))
            if self.num_max_mix > 2:
                noise_sp_inp_2 = torch.FloatTensor(np.array(padded_ns_sp_2))

        if self.return_16k_gt:
            gt_sp16k = torch.FloatTensor(np.array(padded_gt_16k))
            sp_16k_len = torch.IntTensor(sp_16k_len)
            if self.return_noise:
                if self.num_max_mix > 2:
                    return mixed_sp_inp, gt_sp_inp, ctx_inp, ctx_pad, sp_len, f_names, gt_sp16k, sp_16k_len, noise_sp_inp_1, noise_sp_inp_2
                else:
                    return mixed_sp_inp, gt_sp_inp, ctx_inp, ctx_pad, sp_len, f_names, gt_sp16k, sp_16k_len, noise_sp_inp_1
            else:
                return mixed_sp_inp, gt_sp_inp, ctx_inp, ctx_pad, sp_len, f_names, gt_sp16k, sp_16k_len
        else:
            if self.return_noise:
                if self.num_max_mix > 2:
                    return mixed_sp_inp, gt_sp_inp, ctx_inp, ctx_pad, sp_len, f_names, noise_sp_inp_1, noise_sp_inp_2
                else:
                    return mixed_sp_inp, gt_sp_inp, ctx_inp, ctx_pad, sp_len, f_names, noise_sp_inp_1
            else:
                return mixed_sp_inp, gt_sp_inp, ctx_inp, ctx_pad, sp_len, f_names

    def collate_fn_no_tok(self, batch):
        # mixed_aud, gt_aud, ctx_txt, os.path.splitext(os.path.basename(f_path))[0]
        sp_len, sp_16k_len, ctxs, f_names = [], [], [], []
        for data in batch:
            sp_len.append(len(data[0]))
            ctxs.append(data[2])
            f_names.append(data[3])
            if self.return_16k_gt:
                sp_16k_len.append(len(data[4]))

        max_sp_len = max(sp_len)
        if self.return_16k_gt:
            max_sp_16k_len = max(sp_16k_len)

        padded_mixed_sp = []
        padded_gt_sp = []
        padded_gt_16k = []
        padded_ns_sp_1 = []
        padded_ns_sp_2 = []

        if self.return_16k_gt:
            if self.return_noise:
                if self.num_max_mix > 2:
                    for mixed_sp, gt_sp, _, _, gt_16k, noise_sp_1, noise_sp_2 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_gt_16k.append(np.concatenate([gt_16k, np.zeros([max_sp_16k_len - len(gt_16k)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
                        padded_ns_sp_2.append(np.concatenate([noise_sp_2, np.zeros([max_sp_len - len(noise_sp_2)])], axis=0))
                else:
                    for mixed_sp, gt_sp, _, _, gt_16k, noise_sp_1 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_gt_16k.append(np.concatenate([gt_16k, np.zeros([max_sp_16k_len - len(gt_16k)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
            else:
                for mixed_sp, gt_sp, _, _, gt_16k in batch:
                    # right padding
                    padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                    padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                    padded_gt_16k.append(np.concatenate([gt_16k, np.zeros([max_sp_16k_len - len(gt_16k)])], axis=0))
        else:
            if self.return_noise:
                if self.num_max_mix > 2:
                    for mixed_sp, gt_sp, _, _, noise_sp_1, noise_sp_2 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
                        padded_ns_sp_2.append(np.concatenate([noise_sp_2, np.zeros([max_sp_len - len(noise_sp_2)])], axis=0))
                else:
                    for mixed_sp, gt_sp, _, _, noise_sp_1 in batch:
                        # right padding
                        padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                        padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))
                        padded_ns_sp_1.append(np.concatenate([noise_sp_1, np.zeros([max_sp_len - len(noise_sp_1)])], axis=0))
            else:
                for mixed_sp, gt_sp, _, _, in batch:
                    # right padding
                    padded_mixed_sp.append(np.concatenate([mixed_sp, np.zeros([max_sp_len - len(mixed_sp)])], axis=0))
                    padded_gt_sp.append(np.concatenate([gt_sp, np.zeros([max_sp_len - len(gt_sp)])], axis=0))

        mixed_sp_inp = torch.FloatTensor(np.array(padded_mixed_sp))
        gt_sp_inp = torch.FloatTensor(np.array(padded_gt_sp))

        sp_len = torch.IntTensor(sp_len)
        if self.return_noise:
            noise_sp_inp_1 = torch.FloatTensor(np.array(padded_ns_sp_1))
            if self.num_max_mix > 2:
                noise_sp_inp_2 = torch.FloatTensor(np.array(padded_ns_sp_2))

        if self.return_16k_gt:
            gt_sp16k = torch.FloatTensor(np.array(padded_gt_16k))
            sp_16k_len = torch.IntTensor(sp_16k_len)
            if self.return_noise:
                if self.num_max_mix > 2:
                    return mixed_sp_inp, gt_sp_inp, ctxs, sp_len, f_names, gt_sp16k, sp_16k_len, noise_sp_inp_1, noise_sp_inp_2
                else:
                    return mixed_sp_inp, gt_sp_inp, ctxs, sp_len, f_names, gt_sp16k, sp_16k_len, noise_sp_inp_1
            else:
                return mixed_sp_inp, gt_sp_inp, ctxs, sp_len, f_names, gt_sp16k, sp_16k_len
        else:
            if self.return_noise:
                if self.num_max_mix > 2:
                    return mixed_sp_inp, gt_sp_inp, ctxs, sp_len, f_names, noise_sp_inp_1, noise_sp_inp_2
                else:
                    return mixed_sp_inp, gt_sp_inp, ctxs, sp_len, f_names, noise_sp_inp_1
            else:
                return mixed_sp_inp, gt_sp_inp, ctxs, sp_len, f_names