
# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=8          # should match with --gres
cpus=64          # should match with --cpus-per-task
quotatype="reserved"

export OMP_NUM_THREADS=4
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
accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/8gpus_1.yaml
seed=42

# export SLURM_JOB_ID=3442208
# unset SLURM_JOB_ID

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="infer" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  infer_script.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/share4v_aug/new/aug_ft_120k \
--dataset_type base \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/aug/new_diffaug/share4v_ft_aug_caps.jsonl \
--batch_size 8 \
--model_path ${model_path} \
--float_type float16  \
--image_rel_path share4v_aug/new/aug_ft_120k \






