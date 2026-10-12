# GIVLA
Official repository for **GIVLA**, NeurIPS 2026

## 📋 Release Plan
We plan to release all resources sequentially after the conference:
- [x] Make this repository public
- [x] Release full source code


## Enviroment setup
Create the Conda environment and install the core dependencies (including the Gaussian head dependency):
``` bash
conda create -n GIVLA python=3.10 -y
cd GIVLA
pip install -r requirements.txt
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation

cd model/depth_anything_v3
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head

```

## Training
Prepare the datasets referenced in dataset/config.yaml to test model performance on LIBERO and LIBERO-plus.
Configure Accelerate first (this is a required step). An example configuration is provided at the end of this section.
We provide the training commands for the two-stage process:
``` bash
accelerate config  # That is very important step.
# stage1
accelerate launch --num_processes 1 --num_machines 1  --use_deepspeed --deepspeed_config_file ds_config.json scripts/finetune_align.py --run_name GIVLA_stage1 --da3_film_in --SF_align  --align_loss_alpha 0.5  --action_head flowmatching --use_augmentation --lr 2.5e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 8 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 1000 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head  --finetune_dpt_linear_film  --finetune_SF_align --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir model_checkpoint/GIVLA/stage_1

# stage2
accelerate launch --num_processes 1 --num_machines 1 --use_deepspeed --deepspeed_config_file ds_config.json scripts/finetune_align.py --run_name GIVLA_stage2 --da3_film_in  --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 8 --image_size 448 --max_steps 150000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head  --finetune_dpt_linear_film --finetune_da3 --finetune_vit --finetune_llm_backbone --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir model_checkpoint/GIVLA/stage_2 --resume  --resume_pretrain --resume_path model_checkpoint/GIVLA/stage_1/step_4000

```
### Accelerate Config Example
Below is an example setup for single-GPU, single-node training with DeepSpeed. Run accelerate config and answer the prompts accordingly:
```
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1  

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no   

Do you wish to optimize your script with torch dynamo?[yes/NO]:no    

Do you want to use DeepSpeed? [yes/NO]: yes                  

Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes

Please enter the path to the json DeepSpeed config file: 
/path/you/want/to/save/this/json/file.json

Do you want to enable deepspeed.zero.Init when using ZeRO Stage-3 for constructing massive models? [yes/NO]: no  

Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]: no

How many GPU(s) should be used for distributed training? [1]:1      
```

## Evaluation
### LIBERO
Set up the LIBERO test environment as follows:
``` bash
conda create -n libero python=3.8.13 -y
conda activate libero
cd LIBERO_Evaluation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
pip install websockets
pip install huggingface_hub
```
Then, start the evaluation server in the GIVLA environment:

``` bash
conda activate GIVLA
cd GIVLA
python scripts/GIVLA_server.py
```
In a separate terminal, run the LIBERO client:
``` bash
conda activate libero
cd LIBERO_evaluation
python libero_client_4tasks.py
```
### LIBERO-plus
The test procedure for LIBERO-plus is almost the same. Please install LIBERO-plus instead of the original LIBERO and adjust the environment setup accordingly based on their official instructions.
