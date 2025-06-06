
mme_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/mme/mme_llama_objects.jsonl
mme_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/mme/mme_tokens.jsonl

pope_input_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/pope/pope_captions.jsonl
pope_output_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/benchmarks/analysis/pope/pope_tokens.jsonl

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="infer" --mpi=pmi2  --gres=gpu:0 -n1 --ntasks-per-node=1 -c 8 --kill-on-bad-exit=1 --quotatype=auto  \
python /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/pos_inference/pos_inference.py \
--input_file_path ${pope_input_file} \
--output_file_path ${pope_output_file} \








