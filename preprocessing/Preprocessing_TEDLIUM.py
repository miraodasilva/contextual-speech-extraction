import librosa
import os, glob
from tqdm import tqdm
import soundfile as sf
import shutil

### SET ###
data_dir = 'set_dir_to/TEDLIUM_release-3'
save_dir = 'set_dir_to/TEDLIUM_release-3_segment'
preprocess_path = 'set_dir_to/TEDLIUM_release-3_processed'
###########

# 1. Segment
modes = ['test', 'train', 'dev']
for mode in modes:
    ##### following espnet recipe
    seg_f = os.path.join('./data/TEDLIUM', f'{mode}.orig', 'segments')
    txt_f = os.path.join('./data/TEDLIUM', f'{mode}.orig', 'text')
    segments = {}
    txts = {}
    with open(seg_f, 'r') as txt:
        lines = txt.readlines()
    for l in lines:
        target_f_name, source_f_name, st_time, end_time = l.strip().split()
        if not source_f_name in segments:
            segments[source_f_name] = []
            segments[source_f_name].append([target_f_name, st_time, end_time])
        else:
            segments[source_f_name].append([target_f_name, st_time, end_time])
    with open(txt_f, 'r') as txt:
        lines = txt.readlines()
    for l in lines:
        target_f_name, *words = l.strip().split()
        txts[target_f_name] = ' '.join(words)

    for source_f_name in segments.keys():
        aud, sr = librosa.load(os.path.join(data_dir, 'legacy', f'{mode}', 'sph', source_f_name + '.sph'), sr=16000)
        
        for seg_info in tqdm(segments[source_f_name]):
            target_f_name, st_time, end_time = seg_info
            text = txts[target_f_name]

            target_aud = aud[int(float(st_time) * sr):int(float(end_time) * sr)]

            save_name = os.path.join(save_dir, mode, source_f_name, target_f_name)
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            sf.write(save_name + '.wav', target_aud, samplerate=sr, subtype='PCM_16')
            with open(save_name + '.txt', 'w') as txt:
                txt.write(text)

# 2. Copy Train

data_path = save_dir

with open(os.path.join(os.path.join(data_dir, 'speaker-adaptation'), 'train.lst'), 'r') as txt:
    split_files = txt.readlines()
    
for split_file in tqdm(split_files):
    split_file = split_file.strip()
    files = glob.glob(os.path.join(data_path, '*', split_file, '*.wav'))
    for f in files:             
        f_name = f'{os.sep}'.join(os.path.normpath(f).split(os.sep)[-2:])
        save_name = os.path.join(preprocess_path, 'train', f_name)
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        shutil.copy(f, save_name)

# 3. Context Making for train

file_path = save_dir

files = sorted(glob.glob(os.path.join(os.path.join(preprocess_path, 'train'), '*', '*.wav')))

for ll, file in enumerate(tqdm(files)):
    dialog_num, f_name = os.path.normpath(file).split(os.sep)[-2:]

    context_save_name = file[:-4] + '.txt'
    os.makedirs(os.path.dirname(context_save_name), exist_ok=True)

    if not os.path.exists(context_save_name):
        paths = os.path.join(file_path, '*', dialog_num, '*.txt')
        contxts = sorted(glob.glob(paths))
    
        current_id = glob.glob(os.path.join(file_path, '*', dialog_num, f_name[:-4] + '.txt'))[0]
    
        ctx_id = contxts.index(current_id)

        # very inefficient way                                                                                  
        con_text = ""
        for kk, contxt in enumerate(contxts[:ctx_id]):
            try:
                with open(contxt, 'r') as txt:
                    line = txt.readlines()[0].strip()
            except:
                line = ' '
            con_text += f'{line}\n'
        with open(context_save_name, 'w') as txt:
            txt.write(con_text)