
# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=0        # should match with --gres
cpus=16          # should match with --cpus-per-task
quotatype="reserved"

source ~/.bashrc
source ~/anaconda3/bin/activate smoe


language=ar

AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export HF_ENDPOINT=https://hf-mirror.com

code_home=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/
cd $code_home


word_association=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/multilingual_data/${language}/relation.json
safe_words=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/multilingual_data/${language}/safe_words.txt
annotation=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/multilingual_data/${language}/annotations.json
inference_data=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/multilingual_data/${language}/response/llava_origin.json

metrics=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/data/metrics.txt
evaluation_type=g

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="eval_amber" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/tools/lmms-eval/lmms_eval/other_evals/AMBER/multilingual_inference.py \
--word_association ${word_association} \
--safe_words ${safe_words} \
--annotation ${annotation} \
--metrics ${metrics} \
--evaluation_type ${evaluation_type} \
--inference_data ${inference_data} \
--language ${language} 