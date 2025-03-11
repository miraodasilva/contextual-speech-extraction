import librosa
import soundfile as sf
from tqdm import tqdm
import os, glob

### SET ###
DailyTalk_path = 'dir_to/DailyTalk/data'
target_path = 'dir_to/DailyTalk_processed_16k'
###########

dialog_file = './data/DailyTalk/train_dialog.txt'

# 1. Resampling
train_audio_sets = []
train_dialogs = []
with open(os.path.join(dialog_file), 'r') as txt:
    lines = txt.readlines()
for l in lines:
    train_dialogs.append(l.strip())
for dialog in train_dialogs:
    dialog_path = os.path.join(DailyTalk_path, dialog)
    files = sorted(glob.glob(os.path.join(dialog_path, '*.wav')), key=lambda x: int(os.path.basename(x.split('_')[0])))
    train_audio_sets.extend(files)

for f in tqdm(train_audio_sets):
    audio, sr = librosa.load(f, sr=16000)
    assert sr == 16000
    save_path = f.replace(DailyTalk_path, target_path + '/train')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    sf.write(save_path, audio, samplerate=sr, subtype='PCM_16')

# 2. Context Making
with open(dialog_file, 'r') as txt:
    lines = txt.readlines()
for ll, line in enumerate(tqdm(lines)):
    dialog = line.strip()
    context_files = sorted(glob.glob(os.path.join(DailyTalk_path, dialog, '*.txt')), key=lambda x: int(os.path.basename(x).split('_')[0]))
    for ctf in context_files:
        context_save_name = os.path.join(target_path, 'train', dialog, os.path.basename(ctf))
        conv_num = int(os.path.basename(ctf).split('_')[0])
        if conv_num == 0:
            with open(context_save_name, 'w') as txt:
                txt.write('')
        else:
            context = []
            # very inefficient way
            for cn in range(conv_num):
                context_file = glob.glob(os.path.join(DailyTalk_path, dialog, f'{cn}_*.txt'))
                assert len(context_file) == 1, f'Something went wrong {len(context_file)}'
                with open(context_file[0], 'r') as txt:
                    context.append(txt.readlines()[0].strip())
            with open(context_save_name, 'w') as txt:
                for ctx in context:
                    txt.write(ctx + '\n')