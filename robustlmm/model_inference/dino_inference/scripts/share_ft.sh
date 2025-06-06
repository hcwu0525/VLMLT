
# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=8         # should match with --gres
cpus=64         # should match with --cpus-per-task
quotatype="reserved"

export OMP_NUM_THREADS=8
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address
# export HF_ENDPOINT=https://hf-mirror.com

export PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/bin:$PATH

# virtual environment
source ~/.bashrc
source ~/anaconda3/bin/activate lt
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/dino_inference
cd $code_base

model_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/grounding-dino-base
accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/8gpus_1.yaml
seed=55

# export SLURM_JOB_ID=3458700
# unset SLURM_JOB_ID

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="infer" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  groundingDINO_inference.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_llama_objects.jsonl \
--dataset_type sharegpt4v_infer \
--image_dir_path /mnt/petrelfs/share_data/quxiaoye/sharegpt4v/data \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_dino_objects.jsonl \
--token_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_tokens.jsonl \
--batch_size 16 \
--model_path ${model_path} 



# export SLURM_JOB_ID=3273170


