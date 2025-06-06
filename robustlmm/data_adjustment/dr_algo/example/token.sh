source ~/.bashrc
source ~/anaconda3/bin/activate lt

code_base=/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/data_adjustment/dr_algo
cd $code_base

llama_obj_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_objects.jsonl
token_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_tokens.jsonl
what_word_file=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_what_word.jsonl
reverse_index_prefix=/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/reverse_index

token_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_tokens.jsonl"
object_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_dinoobj.jsonl"
what_word_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_whatword.jsonl"
reverse_index_prefix="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index"

mkdir -p ${reverse_index_prefix}

function_type=token

python reverse_indexing.py \
--input_path  $token_file \
--output_path ${reverse_index_prefix}/${function_type}_reverse_index.jsonl \
--function $function_type \
--id_key new_idx