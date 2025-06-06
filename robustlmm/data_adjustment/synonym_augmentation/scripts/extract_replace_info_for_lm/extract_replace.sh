source ~/.bashrc
source ~/anaconda3/bin/activate lt

# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

# accelerate_config_file=/mnt/petrelfs/songmingyang/.config/accelerate/8gpus_1.yaml
code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/data_adjustment/synonym_augmentation
cd $code_base

# export SLURM_JOB_ID=3442214
unset SLURM_JOB_ID

gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="diffusion" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./replace_synonym.py \
--function extract_replace_info \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/meta_clip/aug/token_aug/llava_ft_to_aug.json \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/meta_clip/aug/token_aug/syn_replace/extract_replace_info_for_lm/replace_info.jsonl \
--synonym_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/statistics/token_syn.jsonl \
--distribution_reverse_index_files \
"/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_token_reverse_index.jsonl" \
"/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_dino_stat_reverse_index.jsonl" \
"/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_co_occurrence_reverse_index.jsonl" \
"/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_what_word_reverse_index.jsonl" \
--threads 32

