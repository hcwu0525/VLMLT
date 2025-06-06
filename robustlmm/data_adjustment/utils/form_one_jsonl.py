from mhr.utils.utils import append_jsonl,process_jsonl
from tqdm import tqdm
import logging
import os
import argparse


logger = logging.getLogger(__name__)

def process_object_item(object_item):
    if isinstance(object_item,list):
        return object_item
    elif isinstance(object_item, str):
        return object_item.strip().replace("'","").replace("\\","").replace('"',"").split(",")

def resume_from_ckpt(ckpt_file_path):
    if os.path.exists(ckpt_file_path):
        logger.info(f"checkpoint detected! Resume from file: {ckpt_file_path}")
        cache = process_jsonl(ckpt_file_path)
        processed_id = {item["id"]:True for item in cache }
    else:
        os.makedirs(os.path.dirname(ckpt_file_path),exist_ok=True)
        cache = []
        processed_id = {}
    return processed_id

argparser = argparse.ArgumentParser()
argparser.add_argument("--pos_file",type=str,help="input file",default=None)
argparser.add_argument("--llama_file",type=str,help="input file",default=None)
argparser.add_argument("--output_file",type=str,help="output file",default=None)
args = argparser.parse_args()

# pos_file = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_objects_pos.jsonl"
# llama_file = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_objects_llama.jsonl"
# output_file = "/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_statistics.jsonl"
pos_file = args.pos_file
llama_file = args.llama_file
output_file = args.output_file


pos_data = process_jsonl(pos_file)
llama_data = process_jsonl(llama_file)
processed_id = resume_from_ckpt(output_file)
length = len(pos_data)

llama_data_dict = {item["id"]:item for item in llama_data}
for item in tqdm(pos_data):
    if processed_id.get(item["id"],False):
        continue
    llama_item = llama_data_dict.get(item["id"],None)
    if llama_item is None:
        print("llama item not found")
    llama_objects = process_object_item(llama_item["objects"])
    object_dict = dict(llama=llama_objects,pos=process_object_item(item["objects"]))
    item["objects"] = object_dict
    append_jsonl(item,output_file)