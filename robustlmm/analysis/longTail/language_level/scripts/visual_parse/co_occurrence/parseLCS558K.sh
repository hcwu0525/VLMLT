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

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level
cd $code_base

export SLURM_JOB_ID=3077918
unset SLURM_JOB_ID       

database_name=lcs558kvisualobj

gpus=0
cpus=64
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="${database_name}" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./co_occurrence_count.py \
--object_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_dino_stat.jsonl  \
--verbosity info \
--num_workers ${cpus} \
--database_name ${database_name} \
--dataset_uri neo4j://10.140.54.10:7687 \
--username neo4j \
--password "12345678" \
--ckpt_file_path /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level/scripts/visual_parse/co_occurrence/ckpt/lcs558kllama.jsonl \
--parser_type visual