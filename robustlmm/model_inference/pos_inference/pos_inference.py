from mhr.utils.utils import load_json_file,process_jsonl,append_jsonl,write_jsonl
import logging
import os

from copy import deepcopy
from transformers import  HfArgumentParser
from tqdm import tqdm
from dataclasses import dataclass, field

import stanza
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Sequence, Optional, List
from datasets import load_dataset


logger = logging.getLogger(__name__)
class POS_Parser():
    def __init__(
        self,
        inference_args,
    ):
        self.inference_args = inference_args
        self.deduplicate = self.inference_args.deduplicate
        self.init_model()
        self.load_data()
    
    def init_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parser=stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma',device=device)
        
    def load_data(self):
        pass
    
    def resume_file(self,result_data):
        pass
    
    def get_clean_str(self,input_str):
        return input_str.strip().replace("\n","").replace("\r","").replace("\t","").replace("'","_").replace('"',"_")
        
    def process_sentence(self,sentence):
        llema_list = []
        doc = self.parser(sentence).to_dict()
        for sentence_parse in doc:
            for word in sentence_parse:
                xpos = word.get("xpos","none")
                if xpos.startswith('NN'):
                    lemma = word.get("lemma",None)
                    if lemma:
                        clean_lemma = self.get_clean_str(lemma)
                        llema_list.append(clean_lemma)
                    else:
                        logger.warning(f"lemma is None in {word}")
        if self.deduplicate:
            llema_list = list(set(llema_list))
        return llema_list
    
    def process_item(self,item):
        sentence = item['sentence']
        id = item['id']
        new_idx = item.get('new_idx',None)
        image = item.get('image',None)
        llema_list = []
        doc = self.parser(sentence).to_dict()
        for sentence_parse in doc:
            for word in sentence_parse:
                xpos = word.get("xpos","none")
                if xpos.startswith('NN'):
                    lemma = word.get("lemma",None)
                    if lemma:
                        clean_lemma = self.get_clean_str(lemma)
                        llema_list.append(clean_lemma)
                    else:
                        logger.warning(f"lemma is None in {word}")
        if self.deduplicate:
            llema_list = list(set(llema_list))
        output_item = dict(id=id,tokens=llema_list)
        if new_idx:
            output_item['new_idx'] = new_idx
        if image:
            output_item['image'] = image
        self.write_item_to_output_file(output_item)
        return 1
    
    def parallel_process(self,threads=4):
        total = len(self.meta_data)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # 使用线程池并发地执行任务，并创建一个进度条
            results = list(tqdm(executor.map(self.process_item, self.meta_data), total=total))
        return []
    
    # def parallel_process(self,threads=4):
    #     ids = [item["id"] for item in self.meta_data]
    #     images = [item["image"] for item in self.meta_data]
    #     sentences = [item['sentence'] for item in self.meta_data]
    #     new_idxs = [item['new_idx'] for item in self.meta_data]
    #     total = len(sentences)
    #     with ThreadPoolExecutor(max_workers=threads) as executor:
    #         # 使用线程池并发地执行任务，并创建一个进度条
    #         results = list(tqdm(executor.map(self.process_sentence, sentences), total=total))
        
    #     for id,image,token,new_idx in zip(ids,images,results,new_idxs):
    #         item = dict(id=id,image=image,tokens=token,new_idx=str(new_idx))
    #         self.write_item_to_output_file(item)
    #     return []

    def sequential_process(self):
        result_data = []
        for item in tqdm(self.meta_data):
            item_copy = deepcopy(item)
            item_copy['tokens'] = self.process_sentence(item['sentence'])
            item_copy.pop('sentence')
            # result_data.append(item_copy)
            self.write_item_to_output_file(item_copy)
        # self.result_data = result_data
        return []

class POPE_POS_Parser(POS_Parser):
    def load_data(self):
        self.input_file_path = self.inference_args.input_file_path
        raw_data = process_jsonl(self.input_file_path)
        self.meta_data = []
        for item in raw_data:
            self.meta_data.append({"id":item["id"],"sentence":item["output"]})
        
    def write_item_to_output_file(self,item):
        append_jsonl(item,self.inference_args.output_file_path)
        
    def resume_file(self, result_data):
        write_jsonl(result_data,self.inference_args.output_file_path,)

class ShareGPT4VTrainsetPOSParser(POS_Parser):
    def resume_from_ckpt(self):
        if os.path.exists(self.inference_args.output_file_path):
            
            self.cache = process_jsonl(self.inference_args.output_file_path)
            self.processed_id = {item["new_idx"]:1 for item in self.cache}
            if len(self.cache) > 0:
                logger.info(f"checkpoint detected! Resume from file: {self.inference_args.output_file_path}")
                renewed_data = []
                for item in self.meta_data:
                    if self.processed_id.get(item["new_idx"],None) is None:
                        renewed_data.append(item)
                self.meta_data = renewed_data
        else:
            logger.info(f"Output file {self.inference_args.output_file_path} does not exist, start from scratch")
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        self.input_file_path = self.inference_args.input_file_path
        raw_data = load_json_file(self.input_file_path)
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            res_str=""
            for conv in item["conversations"]:
                # if conv["from"] == "gpt":
                    res_str += f"{conv['value']}\n"
            image_path = item.get("image",None)
            self.meta_data.append({"id":item["id"],"image":image_path,"sentence":res_str,"new_idx":str(idx)})
        self.resume_from_ckpt()
            
    def write_item_to_output_file(self,item):
        if item.get("image",None):
            target_item = dict(id=item["id"],image=item["image"],tokens=item["tokens"],new_idx=item["new_idx"])
        else:
            target_item = dict(id=item["id"],tokens=item["tokens"],new_idx=item["new_idx"])
        append_jsonl(target_item,self.inference_args.output_file_path)

class LLaVATrainsetParser(ShareGPT4VTrainsetPOSParser):
    def load_data(self):
        self.input_file_path = self.inference_args.input_file_path
        raw_data = load_json_file(self.input_file_path)
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            new_idx = item["new_idx"]
            res_str=""
            for conv in item["conversations"]:
                # if conv["from"] == "gpt":
                    res_str += f"{conv['value']}\n"
            image_path = item.get("image",None)
            self.meta_data.append({"id":item["id"],"image":image_path,"sentence":res_str,"new_idx":new_idx})
        self.resume_from_ckpt()

class VQAV2POSParser(POS_Parser):
    def resume_from_ckpt(self):
        if os.path.exists(self.inference_args.output_file_path):
            
            self.cache = process_jsonl(self.inference_args.output_file_path)
            self.processed_id = {item["id"]:1 for item in self.cache}
            if len(self.cache) > 0:
                logger.info(f"checkpoint detected! Resume from file: {self.inference_args.output_file_path}")
                renewed_data = []
                for item in self.meta_data:
                    if self.processed_id.get(item["id"],None) is None:
                        renewed_data.append(item)
                self.meta_data = renewed_data
        else:
            logger.info(f"Output file {self.inference_args.output_file_path} does not exist, start from scratch")
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        self.input_file_path = self.inference_args.input_file_path
        extra_answer_data = load_json_file(self.inference_args.extra_answer_file)['annotations']
        extra_answer_dict = {str(item["question_id"]):item["answers"] for item in extra_answer_data}
        raw_data = load_json_file(self.input_file_path)["questions"]
        self.meta_data = []
        for item in raw_data:
            id = str(item["question_id"])
            answers = extra_answer_dict.get(id,[])
            answers = list(set([item["answer"] for item in answers]))
            ans_str=item["question"]
            for idx,ans in enumerate(answers):
                ans_str+=" "+ans
            self.meta_data.append({"id":id,"sentence":ans_str})
        
    def write_item_to_output_file(self,item):
        append_jsonl(item,self.inference_args.output_file_path)


class SQAPOSParser(POS_Parser):
    def resume_from_ckpt(self):
        if os.path.exists(self.inference_args.output_file_path):
            self.cache = process_jsonl(self.inference_args.output_file_path)
            self.processed_id = {item["id"]:1 for item in self.cache}
            if len(self.cache) > 0:
                logger.info(f"checkpoint detected! Resume from file: {self.inference_args.output_file_path}")
                renewed_data = []
                for item in self.meta_data:
                    if self.processed_id.get(item["id"],None) is None:
                        renewed_data.append(item)
                self.meta_data = renewed_data
        else:
            logger.info(f"Output file {self.inference_args.output_file_path} does not exist, start from scratch")
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        self.input_file_path = self.inference_args.input_file_path
        raw_data = load_dataset(self.input_file_path,self.inference_args.extra_subset_suffix)["test"]
        
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            question = item["question"]
            choices = " ".join(item["choices"])
            lecture = item["lecture"]
            solution = item["solution"]
            sentence = f"{question} {choices} {lecture} {solution}"
            self.meta_data.append({"id":str(idx),"sentence":sentence})
        self.resume_from_ckpt()
            
    def write_item_to_output_file(self,item):
        target_item = dict(id=item["id"],tokens=item["tokens"])
        append_jsonl(target_item,self.inference_args.output_file_path)

inference_model_dict = {
    "pope_parse_tokens_from_caption": POPE_POS_Parser,
    "sharegpt4v_token_parse": ShareGPT4VTrainsetPOSParser,
    "llava_token_parse": LLaVATrainsetParser,
    "vqav2_token_parse": VQAV2POSParser,
    "sqa_token_parse": SQAPOSParser,
    "base": POS_Parser
}
        
if __name__ == "__main__":
        
    @dataclass
    class InferenceArguments:
        input_file_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k.json")
        output_file_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k_objects_llm.jsonl")
        threads: int = field(default=1)
        inference_type: str = field(default="pope_parse_tokens_from_caption")
        deduplicate: bool = field(default=False)
        extra_answer_file : Optional[str] = field(default=None)
        extra_subset_suffix : Optional[str] = field(default="ScienceQA-FULL")
        
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]
    parser = inference_model_dict[args.inference_type](args)
    if args.threads > 1:
        results = parser.parallel_process(threads=args.threads)
    else:
        results = parser.sequential_process()
    # parser.resume_file(results)