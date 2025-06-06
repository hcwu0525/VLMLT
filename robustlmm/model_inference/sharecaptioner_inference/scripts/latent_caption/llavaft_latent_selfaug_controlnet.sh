
# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=4           # should match with --gres
cpus=32          # should match with --cpus-per-task
quotatype="spot"

export OMP_NUM_THREADS=8
AD_NAME=songmingyang
encrypted_password=MJKtQ7daXckx7ujtpVTu6ICPHEc6i2kGq4IQk3tNq1wRWQMnNSR6XoKZoOpf
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address
# export HF_ENDPOINT=https://hf-mirror.com

export PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/bin:$PATH

# virtual environment
source ~/.bashrc
source ~/anaconda3/bin/activate smoe
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/sharecaptioner_inference
cd $code_base

# model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/ShareCaptioner
accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/4gpus_2.yaml
seed=42

# export SLURM_JOB_ID=3442208
# unset SLURM_JOB_ID

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="infer" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  infer_script.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/latent_aug/llavaft_selfaug_control \
--dataset_type base \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/latent/llavaft/latent_reps/latent_supply/self_aug/control_sharecaptioner_caps.jsonl \
--batch_size 8 \
--model_path ${model_path} \
--float_type float16  \
--image_rel_path latent_aug/llavaft_selfaug_control \






