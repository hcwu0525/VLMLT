# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate vllm


export SLURM_JOB_ID=3442214
# unset SLURM_JOB_ID     

gpus=8
cpus=64
quotatype="reserved"

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/llama_inference
cd $code_base

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./llama_inference.py \
--model_path /mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/meta_clip/aug/token_aug/llava_ft_to_aug.json \
--synonym_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/statistics/token_syn.jsonl \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reformed_data/meta_clip/aug/token_aug/llava_ft_token_aug.jsonl  \
--function token_augmentation \
--inference_type token_augment \
--max_tokens 2048
