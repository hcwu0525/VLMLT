# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate lt


AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export HF_ENDPOINT=https://hf-mirror.com

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/vision_level
cd $code_base

export SLURM_JOB_ID=3135332 
unset SLURM_JOB_ID     

gpus=8
cpus=64
quotatype="reserved"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="DINO_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch --config_file /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/vision_level/scripts/accelerate/8gpus.yaml parse_vision_objects.py \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_statistics.jsonl \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_dino_stat.jsonl \
--image_dir_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/lcs558k \
--model_path /mnt/petrelfs/songmingyang/songmingyang/model/mm/grounding-dino-base \
--batch_size 32
