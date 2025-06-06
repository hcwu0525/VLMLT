# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate vllm


export SLURM_JOB_ID=3273170
# unset SLURM_JOB_ID     

gpus=8
cpus=64
quotatype="reserved"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_LCS558K" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level/llama_inference/llama_inference.py \
--model_path /mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/mme \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/mme/mme_llama_objects.jsonl \
--function sentence_extract_object \
--inference_type pope_parse_objects_from_caption
