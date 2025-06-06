source ~/.bashrc
source ~/anaconda3/bin/activate lt


code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level
cd $code_base

export SLURM_JOB_ID=3077918
unset SLURM_JOB_ID       

database_name=sharegpt4vptvisualobj

gpus=0
cpus=64
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="${database_name}" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./co_occurrence_count.py \
--object_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/sharegpt4v_pt_llama_objects.jsonl  \
--verbosity info \
--num_workers ${cpus} \
--database_name ${database_name} \
--dataset_uri neo4j://10.140.54.16:7687 \
--username neo4j \
--password "12345678" \
--ckpt_file_path /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level/scripts/visual_parse/co_occurrence/ckpt/sharegpt4v_pt.jsonl \
--parser_type base


