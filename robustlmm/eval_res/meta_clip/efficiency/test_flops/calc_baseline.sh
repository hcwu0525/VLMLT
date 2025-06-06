source ~/.bashrc
source ~/anaconda3/bin/activate smoe

unset SLURM_JOB_ID

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/eval_res/dr_algo/efficiency/test_flops
cd $code_base

model_path=/mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/Qwen2-VL-7B-Instruct
input_path=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/llama_factory/lt/llava/part_of_data/llava_v1_5_mix665k.json
output_path=./res/baseline_flops.jsonl

gpus=1
cpus=16
quotatype="auto"

OMP_NUM_THREADS=1 srun --partition=MoE --job-name="eval" --mpi=pmi2 \
--gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python ./calc_flops.py \
--model_path ${model_path} \
--input_path  ${input_path} \
--output_path ${output_path}
