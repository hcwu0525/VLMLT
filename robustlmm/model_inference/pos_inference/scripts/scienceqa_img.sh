# unset SLURM_JOB_ID
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



sqa_file=/mnt/petrelfs/songmingyang/songmingyang/data/mm/hf-benchmarks/lmms-lab-ScienceQA
sqa_full_output_path=/mnt/petrelfs/songmingyang/songmingyang/data/mm/hf-benchmarks/lmms-lab-ScienceQA/ScienceQA-IMG/statistics/sqafull_tokens.jsonl
extra_subset_suffix='ScienceQA-IMG'

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/pos_inference/pos_inference.py \
--input_file_path ${sqa_file} \
--output_file_path ${sqa_full_output_path} \
--inference_type sqa_token_parse \
--extra_subset_suffix ${extra_subset_suffix} \
--threads 16




