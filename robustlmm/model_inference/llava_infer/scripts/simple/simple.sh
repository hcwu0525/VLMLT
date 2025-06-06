
# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=4        # should match with --gres
cpus=32         # should match with --cpus-per-task
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="infer" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/llava_infer/simpleinfer.py






