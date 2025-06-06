
# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=1          # should match with --gres
cpus=8          # should match with --cpus-per-task
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
source ~/anaconda3/bin/activate lt
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/dino_inference
cd $code_base

model_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/grounding-dino-base
accelerate_config_file=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/configs/1gpu_2.yaml
seed=55

export SLURM_JOB_ID=3230496
unset SLURM_JOB_ID

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="infer" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file=${accelerate_config_file}  groundingDINO_inference.py \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/pope/pope_llama_objects.jsonl \
--dataset_type pope_infer \
--coco_path /mnt/petrelfs/songmingyang/quxiaoye/VCD_file/val2014 \
--gqa_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/gqa/images \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/pope/dino_pope_objects.jsonl \
--batch_size 32 \
--model_path ${model_path} 






