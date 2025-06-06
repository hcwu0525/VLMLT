source ~/.bashrc
source ~/anaconda3/bin/activate control

# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/8gpus_1.yaml
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/controlnet_inference/inference
cd $code_base

# export SLURM_JOB_ID=3304670

gpus=8
cpus=64
quotatype="auto"
OMP_NUM_THREADS=${gpus} srun --partition=MoE --job-name="control" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  ./controlnet_inference.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/statistics/llavaft_toaug.jsonl \
--img_dir_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/llava_aug/controlaug \


