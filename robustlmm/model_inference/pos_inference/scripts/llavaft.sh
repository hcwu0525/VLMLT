unset SLURM_JOB_ID
source ~/.bashrc
source ~/anaconda3/bin/activate smoe

AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

gpus=1
cpus=16
quotatype="reserved"



mme_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/mme/mme_llama_objects.jsonl
mme_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/mme/mme_tokens.jsonl

pope_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/pope/pope_captions.jsonl
pope_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/pope/pope_tokens.jsonl

sharegpt4v_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
sharegpt4v_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_tokens.jsonl

llava_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llava_v1_5_mix665k_newidx.json
llava_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_tokens.jsonl

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/pos_inference/pos_inference.py \
--input_file_path ${llava_input_file} \
--output_file_path ${llava_output_file} \
--inference_type sharegpt4v_token_parse \
--threads 16


# salloc --partition=MoE --job-name="shutdown"  -c 64 -w SH-IDCA1404-10-140-54-89




