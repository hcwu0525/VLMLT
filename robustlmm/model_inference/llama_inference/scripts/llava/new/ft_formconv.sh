# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate vllm


# export SLURM_JOB_ID=3352621
# unset SLURM_JOB_ID     

gpus=4
cpus=32
quotatype="spot"

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/llama_inference
cd $code_base

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./llama_inference.py \
--model_path /mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/aug/llavaft_augcap.jsonl \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/aug/llavaft_augconv.jsonl \
--function llava_caption_to_conversation \
--inference_type plainaug_caption2instance \
--max_tokens 1024 \
--batch_size 32 \
--tensor_parallel_size 4 