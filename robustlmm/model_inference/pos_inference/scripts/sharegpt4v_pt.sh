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

sharegpt4v_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/share-captioner_coco_lcs_sam_1246k_1107.json
sharegpt4v_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/sharegpt4v_tokens.jsonl


OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/pos_inference/pos_inference.py \
--input_file_path ${sharegpt4v_input_file} \
--output_file_path ${sharegpt4v_output_file} \
--inference_type sharegpt4v_token_parse \
--threads 16







