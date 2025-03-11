import os, glob
import json
from tqdm import tqdm
import subprocess
import shutil

### SET ###
source_path = 'dir_to/SpokenWoz'
segment_path = 'dir_to/SpokenWoz_segment_16k'
Preprocessed_path = 'dir_to/SpokenWoz_processed_16k'
###########

modes = ['train_dev', 'test']
tags = {'user': '0', 'system': '1'}

# 1. Segment
for mode in modes:
    with open(os.path.join(source_path, f'text_5700_{mode}/data.json')) as fd:
         json_data = json.load(fd)
    for file in tqdm(json_data):
        dialogs = json_data[file]['log']
        for turn, dialog in enumerate(dialogs):
            transcript = dialog['text']
            start_time = dialog['words'][0]['BeginTime']
            end_time = dialog['words'][-1]['EndTime']
            duration = end_time - start_time
            tag = dialog['tag']
            
            save_file_name = os.path.join(segment_path, mode, file, f'{turn}_{tags[tag]}_{file}.wav')
            if not os.path.exists(os.path.dirname(save_file_name)):
                os.makedirs(os.path.dirname(save_file_name))

            start_time = start_time / 1000.
            duration = duration / 1000.
            subprocess.call(f"ffmpeg -loglevel panic -nostdin -y -ss {start_time} -i {os.path.join(source_path, f'audio_5700_{mode}', f'{file}.wav')} -t {duration} -acodec pcm_s16le -ar 16000 -ac 1 {save_file_name}", shell=True)

            with open(save_file_name[:-4] + '.txt', 'w') as txt:
                txt.write(transcript + '\n')
                txt.write(f'{duration:.5f}' + '\n')

# 2. Copy Train
val_audio_sets = []
val_dialogs = []
with open(os.path.join(source_path, f'text_5700_train_dev/valListFile.json'), 'r') as txt:
    lines = txt.readlines()
for l in lines:
    val_dialogs.append(l.strip())
    
data_path = segment_path

files = glob.glob(os.path.join(data_path, 'train_dev', '*', '*.wav'))
for file in tqdm(files):
    dialog_name, f_name = os.path.split(file)[-2:]
    if dialog_name not in val_dialogs:
        target_path = os.path.join(Preprocessed_path, 'train', dialog_name, f_name)
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))
        shutil.copy(file, target_path)
    else:
        continue

# 3. Context Making

dialogs = sorted(os.listdir(os.path.join(Preprocessed_path, 'train')))
for dialog in tqdm(dialogs):
    context_files = sorted(glob.glob(os.path.join(data_path, 'train_dev', dialog, '*.txt')), key=lambda x: int(os.path.basename(x).split('_')[0]))
    for ctf in context_files:
        context_save_name = os.path.join(Preprocessed_path, 'train', dialog, os.path.basename(ctf))
        conv_num = int(os.path.basename(ctf).split('_')[0])
        if conv_num == 0:
            with open(context_save_name, 'w') as txt:
                txt.write('')
        else:
            context = []
            # very inefficient way
            for cn in range(conv_num):
                context_file = glob.glob(os.path.join(data_path, 'train_dev', dialog, f'{cn}_*.txt'))
                assert len(context_file) == 1, f'Something went wrong {len(context_file)}'
                with open(context_file[0], 'r') as txt:
                    context.append(txt.readlines()[0].strip())
            with open(context_save_name, 'w') as txt:
                for ctx in context:
                    txt.write(ctx + '\n')