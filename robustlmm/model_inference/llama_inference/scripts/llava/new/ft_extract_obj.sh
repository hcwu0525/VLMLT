# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate vllm


# export SLURM_JOB_ID=3458700
# unset SLURM_JOB_ID     

gpus=8
cpus=64
quotatype="reserved"

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/llama_inference
cd $code_base

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./llama_inference.py \
--model_path /mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llava_v1_5_mix665k_newidx.json \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_llamaobj.jsonl  \
--function existence_conversation \
--inference_type llava_parse_objects \
--max_tokens 256
