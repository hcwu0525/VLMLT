from mhr.utils.utils import *
from tqdm import tqdm

from transformers import  HfArgumentParser
from tqdm import tqdm
from dataclasses import dataclass, field


# def str2list(input_str):
#     if isinstance(input_str,str):
#         raw_list = input_str.strip().replace("\n","").split(",")
#         new_list = []
#         for item in raw_list:
#             new_list.append(item.strip())
#         return new_list
#     elif isinstance(input_str,list):
#         return input_str
#     else:
#         raise TypeError("input_str should be str or list")

# def get_two_words(word1,word2):
#     if word1 < word2:
#         return f"{word1},{word2}"
#     else:
#         return f"{word2},{word1}"
    

origin_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/share-captioner_coco_lcs_sam_1246k_1107.json"
token_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/sharegpt4v_tokens.jsonl"
object_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/sharegpt4v_pt_dino_objects.jsonl"
what_word_file=None

origin_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
token_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_tokens.jsonl"
# object_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_dino_objects.jsonl"
object_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_dino_objects.jsonl"
what_word_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/sharegpt4v_ft_llama_what_words.jsonl"

# origin_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k.json"
# token_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_tokens.jsonl"
# object_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_objects.jsonl"

# output_path = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/origin_data_full_info/share_ft_dinoobj.json"
# output_path = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/statistics/llavapt_fullinfo.json"
output_path="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/origin_data_full_info/share_pt_llamaobj.json"

origin_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llava_v1_5_mix665k_newidx.json"
token_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_tokens.jsonl"
object_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_dinoobj.jsonl"
what_word_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_whatword.jsonl"
output_path="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/llavaft_fullinfo.json"

origin_data = load_json_file(origin_file)
token_data = process_jsonl(token_file)
object_data = process_jsonl(object_file)
what_data = process_jsonl(what_word_file)

id_key = "new_idx"

object_to_item_dict = {str(item[id_key]):str2list(item.get("objects",item.get("outputs",""))) for item in object_data}
token_to_item_dict = {str(item[id_key]):str2list(item["tokens"]) for item in token_data}
what_word_to_item_dict = {str(item[id_key]):str2list(item.get("objects",item.get("outputs",""))) for item in what_data}

def build_full_info_file():
    output_data=[]
    for idx,item in enumerate(tqdm(origin_data)):
        # new_idx = str(idx)
        id = str(item["new_idx"])
        # id = new_idx
        tokens = token_to_item_dict.get(str(id),[])
        objects = object_to_item_dict.get(str(id),[])
        co_occurrence = objects_to_co_occurrence(objects)
        what_words = what_word_to_item_dict.get(str(id),[])
        statistics = dict(token=tokens,object=objects,co_occurrence=co_occurrence,what_word=what_words)
        item["statistics"]=statistics
        # item["new_idx"] = new_idx
        output_data.append(item)
    return output_data

def objects_to_co_occurrence(objects):
    co_occurrence = []
    objects = list(set(objects))
    if len(objects) < 2:
        return []
    for i in range(len(objects)):
        for j in range(i+1,len(objects)):
            co_occurrence.append(get_two_words(objects[i],objects[j]))
    return co_occurrence

if __name__ == "__main__":
    output_data = build_full_info_file()
    write_json_file(output_data,output_path)