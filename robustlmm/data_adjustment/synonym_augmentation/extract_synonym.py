from nltk.corpus import wordnet
from mhr.utils.utils import *

import os

import os
from tqdm import tqdm
import torch
import random
import logging


from dataclasses import dataclass, field
from transformers import HfArgumentParser

class SynonymExtractor:
    def __init__(self,args):
        self.args = args
        self.keyword = args.keyword
        
        self.prepare_token_threshold()
        self.resume_from_ckpt()
        self.load_data()
    
    def prepare_token_threshold(self):
        token_reverse_file = '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_token_reverse_index.jsonl'
        token_reverse_data = process_jsonl(token_reverse_file)
        self.token_threshold = 120
        self.token_reverse_num_dict = {item["object"]:len(item["ids"]) for item in token_reverse_data}
        
        
    def extract_synonyms(self, word):
        synonyms=[]
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                synonyms.append(lm.name())
        return list(set(synonyms))

    def process_one_item(self,item):
        token = item["token"]
        synonyms = self.extract_synonyms(token)
        synonym_dict = []
        for synonym in synonyms:
            thres = self.token_threshold - self.token_reverse_num_dict.get(synonym,0)
            synonym_dict.append({"synonym":synonym,"value":thres})
        synonym_dict = sorted(synonym_dict,key=lambda x:x["value"],reverse=True)
        item["synonyms"] = synonym_dict
        
        self.write_item_into_file(item)

    def load_data(self):
        pass
    
    def write_item_into_file(self,item):
        pass
    
    def resume_from_ckpt(self):
        if os.path.exists(self.args.output_path):
            raw_data = process_jsonl(self.args.output_path)
            self.processed_idx = {}
            for idx, item in enumerate(raw_data):
                self.processed_idx[item["new_idx"]] = 1
        else:
            self.processed_idx = {}
    
    def sequential_process(self):
        for item in tqdm(self.meta_data):
            self.process_one_item(item)
    
    def parallel_process(self):
        pass
                
    
        
class LLaVAFTSynonymExtractor(SynonymExtractor):
    
    def load_data(self):
        token_info = process_jsonl(self.args.input_path)
        self.meta_data = []
        
        token_list=[]
        for item in token_info:
            token_list.extend(item[self.keyword])
        token_list = list(set(token_list))
        idx = 0
        
        for token in token_list:
            if self.processed_idx.get(idx,None) is None:
                self.meta_data.append({"new_idx":idx,"token":token})
            idx += 1
        print_rank0(f"Total number of data to process: {len(self.meta_data)}" )
        
    def prepare_token_threshold(self):
        token_reverse_file = '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_token_reverse_index.jsonl'
        token_reverse_data = process_jsonl(token_reverse_file)
        self.token_threshold = 120
        self.token_reverse_num_dict = {item["object"]:len(item["ids"]) for item in token_reverse_data}
    
    def write_item_into_file(self,item):
        append_jsonl(item,self.args.output_path)

model_dict = dict(
    llava_ft=LLaVAFTSynonymExtractor,
    base=SynonymExtractor,
)

if __name__ == "__main__":
    @dataclass
    class DataArguments:

        input_path: str = field(default_factory=str)
        output_path: str = field(default=None)
        keyword: str = field(default="token")
        # output_log: str = field(default=None)
        seed: int = field(default=42)
        function: str=field(default="llava_ft")
        parallel_mode: str = field(default="sequential")
        
    
    parser = HfArgumentParser((DataArguments))
    args=parser.parse_args_into_dataclasses()[0]
    random.seed(args.seed)
    model = model_dict[args.function](args)
    if args.parallel_mode == "sequential":
        model.sequential_process()
    else:
        model.parallel_process()
    
