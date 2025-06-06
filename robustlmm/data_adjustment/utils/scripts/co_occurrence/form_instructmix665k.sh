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

database_name=instructmix665kllama

gpus=0
cpus=32
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="reform" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./form_one_jsonl.py \
--pos_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_objects_pos.jsonl \
--llama_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_objects_llama.jsonl \
--output_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_statistics.jsonl 
