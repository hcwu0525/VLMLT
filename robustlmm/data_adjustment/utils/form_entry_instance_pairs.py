from mhr.utils.utils import append_jsonl,process_jsonl,write_json_file,load_json_file
from tqdm import tqdm
import logging
import os
import argparse


logger = logging.getLogger(__name__)

def process_object_item(object_item):
    if isinstance(object_item,list):
        return list(set(object_item))
    elif isinstance(object_item, str):
        return object_item.strip().replace("'","").replace("\\","").replace('"',"").split(",")

def resume_from_ckpt(ckpt_file_path):
    if os.path.exists(ckpt_file_path):
        logger.info(f"checkpoint detected! Resume from file: {ckpt_file_path}")
        cache = load_json_file(ckpt_file_path)
        processed_id = {item["id"]:True for item in cache }
    else:
        os.makedirs(os.path.dirname(ckpt_file_path),exist_ok=True)
        cache = []
        processed_id = {}
    return processed_id

def get_two_words(word1,word2):
    if word1 < word2:
        return f"{word1},{word2}"
    else:
        return f"{word2},{word1}"

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--origin_file",type=str,help="input file",default=None)
    argparser.add_argument("--object_file",type=str,help="input file",default=None)
    argparser.add_argument("--token_file",type=str,help="input file",default=None)
    argparser.add_argument("--co_occurrence_file",type=str,help="input file",default=None)
    argparser.add_argument("--what_word_file",type=str,help="input file",default=None)
    argparser.add_argument("--output_file",type=str,help="output file",default=None)
    args = argparser.parse_args()




    object_data = process_jsonl(args.object_file)
    token_data = process_jsonl(args.token_file)
    # co_occurrence_data = process_jsonl(args.co_occurrence_file)
    what_word_data = process_jsonl(args.what_word_file)
    origin_data = load_json_file(args.origin_file)

    object_dict = {item["id"]:item for item in object_data}
    token_dict = {item["id"]:item for item in token_data}
    # co_occurrence_dict = {item["id"]:item for item in co_occurrence_data}
    what_word_dict = {item["id"]:item for item in what_word_data}
    final_data = []
    for item in tqdm(origin_data):
        visual_item = object_dict.get(item["id"],None)
        token_item = token_dict.get(item["id"],None)
        # co_occurrence_item = co_occurrence_dict.get(item["id"],[])
        what_word_item = what_word_dict.get(item["id"],None)
        if visual_item:
            visual_objects = process_object_item(visual_item["statistic"]["labels"])
        else:
            visual_objects = []
        if token_item:
            tokens = process_object_item(token_item["objects"])
        else:
            tokens = []
        if what_word_item:
            what_word = process_object_item(what_word_item["objects"])
        else:
            what_word = []
        co_occurrence = []
        for i in range(len(visual_objects)):
            for j in range(i+1, len(visual_objects)):
                co_occurrence.append(get_two_words(visual_objects[i],visual_objects[j]))
        co_occurrence = list(set(co_occurrence))
        item["statistics"] = {"object":visual_objects,"token":tokens,"co_occurrence":co_occurrence,"what_word":what_word} 
        # append_jsonl(item,args.output_file)
        final_data.append(item)
    write_json_file(final_data,args.output_file)
