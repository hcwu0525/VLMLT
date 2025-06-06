source ~/.bashrc
source ~/anaconda3/bin/activate lt

# environment variables
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export HF_ENDPOINT=https://hf-mirror.com

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level
cd $code_base

# export SLURM_JOB_ID=3098457
unset SLURM_JOB_ID

gpus=0
cpus=1
quotatype="reserved"
OMP_NUM_THREADS=1 srun --partition=MoE --job-name="postgres" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  -w SH-IDCA1404-10-140-54-108 \
createdb testdb

# salloc --partition=MoE --job-name="postgres" --gres=gpu:0 -n1 --ntasks-per-node=1 -c 80 --quotatype="reserved"