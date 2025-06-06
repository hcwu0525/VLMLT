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
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/vqav2/val2014_annotations/captions_val2014.json \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/vqav2/statistics/val14_llama_obj.jsonl  \
--extra_caption_file /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/vqav2/statistics/val14_llava13b_caps.jsonl \
--function sentence_extract_object \
--inference_type val2014_extract \
--max_tokens 256
