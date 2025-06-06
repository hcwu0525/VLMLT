source ~/.bashrc
source ~/anaconda3/bin/activate lt



code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/data_adjustment/dr_algo
cd $code_base

unset SLURM_JOB_ID       

keys=("token" "object" "co_occurrence")
alpha=1
pass_num=1
IFS=' ' compose_list="${keys[*]}"

for str in "${keys[@]}"; do
  file_insert="${file_insert}_${str}"
done

gpus=0
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="reform" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./reform_data_one_scope.py \
--input_dataset_file /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_fullinfo.json \
--output_file  /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/train_data/llava_meta_toc_p${pass_num}_a${alpha}.json \
--mode "compose_alpha" \
--compose_list ${compose_list} \
--pass_num $pass_num \
--alpha ${alpha} \
--target_model llava_pt


