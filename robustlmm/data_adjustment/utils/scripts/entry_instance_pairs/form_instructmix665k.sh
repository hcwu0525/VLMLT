source ~/.bashrc
source ~/anaconda3/bin/activate lt

# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export HF_ENDPOINT=https://hf-mirror.com

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/vision_level/utils
cd $code_base

export SLURM_JOB_ID=3077918
unset SLURM_JOB_ID       


gpus=0
cpus=32
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="reform" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./form_entry_instance_pairs.py \
--origin_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
--object_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_dino_stat.jsonl \
--token_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_objects_pos.jsonl \
--what_word_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_what_word_llama.jsonl \
--output_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/test/llava_v1_5_mix665k.json 

