# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export HF_ENDPOINT=https://hf-mirror.com

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/analysis/longTail/language_level
cd $code_base

export SLURM_JOB_ID=3077918
unset SLURM_JOB_ID     

gpus=0
cpus=32
quotatype="reserved"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="LM_Instruct665k" --mpi=pmi2 -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./lm_parse_dataset.py \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k_objects.jsonl \
--num_workers ${cpus} \
--verbosity info

#   parser.add_argument('--input_file_path', type=str, help='input file path', default="lcs558k")
#     parser.add_argument('--output_file_path', type=str, help='output file path', default="stanza")
#     parser.add_argument("--num_workers",type=int,help="number of workers",default=64)
#     parser.add_argument("--verbosity", type=str, help="verbosity", default="info")