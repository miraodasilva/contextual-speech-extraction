# Contextual Speech Extraction
![image](https://github.com/user-attachments/assets/6a211c31-bfa1-4c27-862a-965deef7a781)

This repository contains the PyTorch implementation of the following paper:
> **Contextual Speech Extraction: Leveraging Textual History as an Implicit Cue for Target Speech Extraction**<br>
> Minsu Kim*, Rodrigo Mira*, Honglie Chen, Stavros Petridis, Maja Pantic (*Equal contribution)  <br>
> \[[Paper](https://arxiv.org/abs/)\]

## Requirements
- python 3.9
- pytorch 2.4.1
- torchvision
- torchaudio
- transformers
- speechbrain
- av
- tensorboard
- openai-whisper
- librosa
- einops
- torchmetrics
- idr_torch

### Datasets
#### Download
DailyTalk
- https://github.com/keonlee9420/DailyTalk

SpokenWoz
- https://spokenwoz.github.io/SpokenWOZ-github.io/

TEDLIUM3
- https://www.openslr.org/51/

DEMAND
- https://www.kaggle.com/datasets/aanhari/demand-dataset

#### Preprocessing
After download the dataset, segment waveform and resample data to 16k Hz. <br>
Also, generate contexture text.

Please refer to [./preprocessing](/preprocessing) dir to get information about the steps.

```
# if the transcription looks like this
0_0_d1.wav: "Well, how does it look?"
1_1_d1.wav: "It's a perfect fit."
2_0_d1.wav: "Let me pay for it now"

# the context data contain all previous dialog history
0_0_d1.txt: " " (Empty as no previous turn)
1_1_d1.txt: "Well, how does it look?"
2_0_d1.txt: "Well, how does it look?\nIt's a perfect fit."
```

We suppose the training data directory is constructed as

```
DailyTalk_processed
├── train
|   ├── 1
|   |   └── *.wav
|   |   └── *.txt
|   ├── *
```

```
SpokenWoz_processed
├── train
|   ├── MUL0003
|   |   └── *.wav
|   |   └── *.txt
```

```
TEDLIUM3_processed
├── train
|   ├── 911Mothers_2010W
|   |   └── *.wav
|   |   └── *.txt
|   ├── *
```

The validation and test sets for each dataset can be found at below. <br>
DailyTalk: Mixed Speech for [Validation & Test](https://drive.google.com/file/d/1Dy0EaJSDHT7rHGnI9pDnbkcMqDW307GH/view?usp=drive_link) <br>
SpokenWoz: Mixed Speech for [Validation & Test](https://drive.google.com/file/d/1nvnwwtwaEulvHT34jX1CtBLkrLXPucJb/view?usp=drive_link) <br>
TEDLIUM3: Mixed Speech for [Validation & Test](https://drive.google.com/file/d/14W-Wnx5795D3cHvX17Wa8IqvZWFoXQ9A/view?usp=drive_link) <br>

Put each `val` and `test` in the directory of each processed data.

## Pretrained Sepformer
Unified CSE models are initialized from Sepformer trained on each dataset. <br>
We provide the pre-trained on Sepformer models. <br>

- [DailyTalk](https://drive.google.com/file/d/1Bkri_XucOszWyS3QjkE7cVgQSRFZV4kg/view?usp=sharing) <br>
- [SpokenWoz](https://drive.google.com/file/d/1x49iKuFyQG-ushcuBhHMbj8wV0ED-Ghu/view?usp=sharing) <br>
- [TEDLIUM3](https://drive.google.com/file/d/1qxsgvd7K3CnWmmQl1yEQ5UPnRgCUUfD1/view?usp=sharing) <br>
- [TEDLIUM3 (3spk)](https://drive.google.com/file/d/1lVmAAfrm3Ksax0ObVRbQcZpvt_y4U5QA/view?usp=sharing)

## Training the Model
To train the model, run following command:

```shell
# Example: ContSep model training using 8 GPUs on SpokenWOZ 2 speaker
GPUS=0,1,2,3,4,5,6,7
Experiment_name=ContSep_SpokenWoz_2spk

num_gpus=$(echo "$GPUS" | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1))

CUDA_VISIBLE_DEVICES=$GPUS \
torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} \
train_ContSep.py \
--dailytalk_data_path `set_dir_to/DailyTalk_processed_16k` \
--spokenwoz_data_path `set_dir_to/SpokenWoz_processed_16k` \
--tedlium_data_path `set_dir_to/TEDLIUM_release-3_processed_16k` \
--acoustic_noise_path `set_dir_to/DEMAND` \
--train_data spokenwoz \
--ctx_weight 5. \
--checkpoint ./pretrained/SpokenWoz_Sepformer.ckpt \
--llama_path meta-llama/Meta-Llama-3-8B \
--llama_auth_token `authorized token` \
--checkpoint_dir ./data/checkpoints/$Experiment_name \
--project $Experiment_name \  # WANDB
--max_sp_len 16 \
--sr 8000 \
--context_length 0 \
--num_max_mix 2 \
--num_test_mix 2 \
--augmentation \
--noise_add \
--speed_perturb_ratio "0.9 1.0 1.1" \
--max_shift_sec 1. \
--max_context_train 150 \
--generate_speech \
--temp_dir ./tmp_eval/$Experiment_name \
--batch_size 2 \
--eval_step 10000 \
--lr 3e-4 \
--gpu $GPUS \
--update_frequency 1 \
--fp16 \
--workers 6 \
--warmup \
--warmup_iteration 10000 \
--tot_iters 500000 \
--masterport 8686 \
--resume \
--torchrun \
--distributed
```

```shell
# Example: ContExt model training using 8 GPUs on DailyTalk 2 speaker
GPUS=0,1,2,3,4,5,6,7
Experiment_name=ContExt_DailyTalk_2spk

num_gpus=$(echo "$GPUS" | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1))

CUDA_VISIBLE_DEVICES=$GPUS \
torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} \
train_ContExt.py \
--dailytalk_data_path `set_dir_to/DailyTalk_processed_16k` \
--spokenwoz_data_path `set_dir_to/SpokenWoz_processed_16k` \
--tedlium_data_path `set_dir_to/TEDLIUM_release-3_processed_16k` \
--acoustic_noise_path `set_dir_to/DEMAND` \
--train_data dailytalk \
--project $Experiment_name \
--checkpoint ./pretrained/DailyTalk_Sepformer.ckpt \
--llama_path meta-llama/Meta-Llama-3-8B \
--llama_auth_token `authorized token` \
--checkpoint_dir ./data/checkpoints/$Experiment_name \
--max_sp_len 16 \
--sr 8000 \
--context_length 0 \
--num_max_mix 2 \
--augmentation \
--noise_add \
--speed_perturb_ratio "0.9 1.0 1.1" \
--max_shift_sec 1. \
--max_context_train 100 \
--generate_speech \
--temp_dir ./tmp_eval/$Experiment_name \
--batch_size 2 \
--eval_step 10000 \
--lr 1.5e-4 \
--gpu $GPUS \
--update_frequency 2 \
--worker 6 \
--warmup \
--warmup_iteration 5000 \
--tot_iters 300000 \
--masterport 8686 \
--resume \
--fp16 \
--torchrun \
--distributed
```

```shell
# Example: HContExt model training using 8 GPUs on TEDLIUM 2 speaker
GPUS=0,1,2,3,4,5,6,7
Experiment_name=HContExt_Tedlium_2spk

num_gpus=$(echo "$GPUS" | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1))

CUDA_VISIBLE_DEVICES=$GPUS \
torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} \
train_HContExt.py \
--dailytalk_data_path `set_dir_to/DailyTalk_processed_16k` \
--spokenwoz_data_path `set_dir_to/SpokenWoz_processed_16k` \
--tedlium_data_path `set_dir_to/TEDLIUM_release-3_processed_16k` \
--acoustic_noise_path `set_dir_to/DEMAND` \
--train_data tedlium \
--project $Experiment_name \
--checkpoint ./pretrained/TEDLIUM_Sepformer.ckpt \
--llama_path meta-llama/Meta-Llama-3-8B \
--llama_auth_token `authorized token` \
--checkpoint_dir ./data/checkpoints/$Experiment_name \
--max_sp_len 16 \
--sr 8000 \
--context_length 0 \
--num_max_mix 2 \
--augmentation \
--noise_add \
--speed_perturb_ratio "0.9 1.0 1.1" \
--max_shift_sec 1. \
--max_context_train 200 \
--generate_speech \
--temp_dir ./tmp_eval/$Experiment_name \
--batch_size 2 \
--eval_step 10000 \
--lr 3e-4 \
--warmup \
--warmup_iteration 10000 \
--gpu $GPUS \
--update_frequency 2 \
--workers 6 \
--tot_iters 500000 \
--masterport 8686 \
--resume \
--fp16 \
--torchrun \
--distributed
```

## Testing the Model
To test the model, run following command:
```shell
# ContSep test example on DailyTalk 2 speaker
GPUS=0
CHECKPOINT=./pretrained/DailyTalk_ContSep.ckpt

CUDA_VISIBLE_DEVICES=$GPUS \
python test.py \
--dailytalk_data_path `/fsx/minsu/Dataset/DailyTalk_processed_16k` \
--spokenwoz_data_path `/fsx/minsu/Dataset/SpokenWoz_preprocessed` \
--tedlium_data_path `/fsx/minsu/Dataset/TEDLIUM_release-3_CSF` \
--test_dataset dailytalk \
--test_model ContSep \
--sr 8000 \
--llama_path meta-llama/Meta-Llama-3-8B \
--llama_auth_token `authorized token` \
--checkpoint $CHECKPOINT \
--max_sp_len 30 \
--context_length 0 \
--num_max_mix 2 \
--num_test_mix 2 \
--batch_size 10 \
--gpu $GPUS \
--fp16 \
--generate_speech \ # if want to save the extracted speech
```

```shell
# H-ContExt test example on Tedlium 2 speaker
GPUS=0
CHECKPOINT=./pretrained/TEDLIUM_HContExt.ckpt

CUDA_VISIBLE_DEVICES=$GPUS \
python test_HContExt.py \
--dailytalk_data_path `/fsx/minsu/Dataset/DailyTalk_processed_16k` \
--spokenwoz_data_path `/fsx/minsu/Dataset/SpokenWoz_preprocessed` \
--tedlium_data_path `/fsx/minsu/Dataset/TEDLIUM_release-3_CSF` \
--test_dataset tedlium \
--sr 8000 \
--cue joint \
--llama_path meta-llama/Meta-Llama-3-8B \
--llama_auth_token `authorized token` \
--checkpoint $CHECKPOINT \
--max_sp_len 30 \
--context_length 0 \
--num_max_mix 2 \
--num_test_mix 2 \
--batch_size 10 \
--gpu $GPUS \
--fp16 \
--generate_speech \ # if want to save the extracted speech
```

```shell
# Cascaded CSE test example on TEDLIUM 3 speaker
GPUS=0
CHECKPOINT=./pretrained/Tedlium_3spk_Sepformer.ckpt

CUDA_VISIBLE_DEVICES=$GPUS \
python test_cascaded.py \
--dailytalk_data_path `/fsx/minsu/Dataset/DailyTalk_processed_16k` \
--spokenwoz_data_path `/fsx/minsu/Dataset/SpokenWoz_preprocessed` \
--tedlium_data_path `/fsx/minsu/Dataset/TEDLIUM_release-3_CSF` \
--test_dataset tedlium \
--sr 8000 \
--llama_path meta-llama/Meta-Llama-3-8B \
--llama_auth_token `authorized token` \
--checkpoint $CHECKPOINT \
--max_sp_len 30 \
--context_length 0 \
--num_max_mix 3 \
--num_test_mix 3 \
--batch_size 1 \
--gpu $GPUS \
--generate_speech \
--fp16
```

## Pre-trained model checkpoints
The pre-trained models are available. <br>

| DailyTalk | SpokenWoz | TEDLIUM3 | TEDLIUM3 (3spks)|
|:--------:|:-------:|:-------:|:-------:|
|[Sepformer](https://drive.google.com/file/d/1Bkri_XucOszWyS3QjkE7cVgQSRFZV4kg/view?usp=sharing) |[Sepformer](https://drive.google.com/file/d/1x49iKuFyQG-ushcuBhHMbj8wV0ED-Ghu/view?usp=sharing) | [Sepformer](https://drive.google.com/file/d/1qxsgvd7K3CnWmmQl1yEQ5UPnRgCUUfD1/view?usp=sharing)|[Sepformer](https://drive.google.com/file/d/1lVmAAfrm3Ksax0ObVRbQcZpvt_y4U5QA/view?usp=sharing) |
|[ContSep](https://drive.google.com/file/d/1vRPs-akm1w27vwxcSy-xqQmaLkwVdF0o/view?usp=sharing) | [ContSep](https://drive.google.com/file/d/1qUPDwWJNNvXOvOAu3zD8MjQcHR0Snu8j/view?usp=sharing)|[ContSep](https://drive.google.com/file/d/1k4QzASDNVHkViQuL6IKThZMVjC_MVQMs/view?usp=sharing) | [ContSep](https://drive.google.com/file/d/11G-3h5uHs-wqQiGj3MENfInhIJUkiyE8/view?usp=sharing)|
|[ContExt](https://drive.google.com/file/d/1Spxv-FH3qSpYR_7aDsegAlNiTTYK6qUL/view?usp=sharing) | [ContExt](https://drive.google.com/file/d/1sGVOfnK09gDdUDgwMUk1sXoKMRK35MT8/view?usp=sharing)| [ContExt](https://drive.google.com/file/d/1_ViM59I_H8DT1OFD67FYcRd8weBBMO_B/view?usp=sharing)|[ContExt](https://drive.google.com/file/d/11HXVh0anOz0SP5dVWMtpz8-WMDaF-ZmW/view?usp=sharing) |
|[H-ContExt](https://drive.google.com/file/d/1LeL29_Z5WQdBhcoldcOphubQtLWHd7I7/view?usp=sharing) |[H-ContExt](https://drive.google.com/file/d/1Bhwz-t24BKrE8duAZncUskoTW2UY4OPF/view?usp=sharing)| [H-ContExt](https://drive.google.com/file/d/14Ir5xFb6brgsPgkpDO1TYh05lRcTbcvG/view?usp=sharing)| [H-ContExt](https://drive.google.com/file/d/1bH3zb5iBerpt7wTxOPTFUWS_0ZvSNyqO/view?usp=sharing)|


## Citation
If you find this work useful in your research, please cite the paper:
```

```
