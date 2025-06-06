
from openai import OpenAI
from copy import deepcopy
from time import sleep
from random import randint
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .prompts import prompt_dict
from mhr.utils.utils import load_json_file, process_jsonl,append_jsonl

import os
import logging

logger = logging.getLogger(__name__)




class BaseLMParser():
    def __init__(self,
                 model_type: str = "deepseek",
                 input_type: str = "existence_conversation",
                 input_dataset_file: str = None,
                 output_file: str = None,
                 model_args : dict = None,
                 model_inference_args: dict = None,
                 ):
        self.input_dataset_file_path = input_dataset_file
        self.output_file_path = output_file
        self.input_type = input_type
        self.model_type = model_type
        self.model_args = model_args
        self.model_inference_args = model_inference_args
        
        self.select_lm()
        self.resume_from_file()
        self.prepare_content_list()
        
    
    
    def select_lm(self):
        if self.model_type == "deepseek":
            self.model_inference_args = {"model":"deepseek-chat","temperature":0,} if self.model_inference_args is None else self.model_inference_args
            self.model_args = dict(api_key="sk-1f91217ff9824e4d914b30d81778fb86", base_url="https://api.deepseek.com") if self.model_args is None else self.model_args
            
            self.prepare_input = self.deepseek_prepare_input
            self.inference = self.deepseek_inference
            self.process_one_item = self.deepseek_process_one_item
            self.messages = self.deepseek_prepare_prompt()
            self.model = self.deepseek_prepare_model(self.model_args)
            logger.info("Deepseek model loaded.")
        elif self.model_type == "llama3":
            pass
            
        else:
            raise ValueError("Model type not supported.")
        
        
    
    def resume_from_file(self):
        assert os.path.exists(self.input_dataset_file_path), f"Input dataset file {self.input_dataset_file_path} does not exist."
        if os.path.exists(self.output_file_path):
            logger.info(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = [item["id"] for item in self.cache]
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = []

    
    def deepseek_prepare_prompt(self):
        if self.input_type == "existence_conversation":
            prompt = prompt_dict[self.input_type]
            messages=[{"role": "system", "content":prompt["system"]}]
            for shot in prompt["fewshot"]:
                messages.append({"role": "user", "content": shot[0]})
                messages.append({"role": "assistant", "content": shot[1]})
            return messages
        
    def deepseek_prepare_input(self,input_content=None):
        if self.messages is not None:
            messages = deepcopy(self.messages)
            messages.append({"role": "user", "content": input_content})
            return messages
        else:
            raise ValueError("No messages provided.")
        
        
            
    def deepseek_inference(self,messages=None,model_inference_args=None,retry=0):
        client = self.model
        try:
            response = client.chat.completions.create(
                messages=messages,
                stream=False,
                **model_inference_args
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(e)
            if retry > 1000:
                raise ValueError("Retry limit exceeded.")
            wait_time = 2**retry + randint(0,10)
            if wait_time > 256:
                wait_time = 256
            logger.info(f"Wait {wait_time} seconds and retry.. ({retry}/1000)")
            sleep(wait_time)
            
            return self.deepseek_inference(messages=messages, model_inference_args=model_inference_args, retry=retry+1)
            
        
    
    def deepseek_process_one_item(self,input_item,):
        item = deepcopy(input_item)
        assert "input_content" in item.keys(), "Input item does not contain input content."
        if item["id"] in self.processed_id:
            return 2
        input_content = item.pop("input_content")
        model_input = self.prepare_input(input_content)
        response = self.inference(model_input,self.model_inference_args)
        item["objects"] = response
        append_jsonl(item,self.output_file_path)
        return 1
    
    def parallel_process(self, num_workers=32):
        logger.info(f"Start processing, num_workers={num_workers}")
        length = len(self.content_list)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(self.process_one_item,self.content_list),total=length))

    def serial_process(self):
        for item in tqdm(self.content_list):
            response = self.process_one_item(item)
    
    def prepare_content_list(self):
        pass
    

class LCS558KLMParser(BaseLMParser):
    def prepare_content_list(self):
        input_list = load_json_file(self.input_dataset_file_path)
        self.content_list=[]
        logger.info(f"Start processing input file: {self.input_dataset_file_path}")
        for item in tqdm(input_list):
            if item["id"] in self.processed_id:
                continue
            conv = item["conversations"]
            res_str = ""
            for shot in conv:
                res_str += f"{shot['from']}: {shot['value']}\n"
            item["input_content"] = res_str
            self.content_list.append(item)
        logger.info(f"Input file process over. Total number of items to process: {len(self.content_list)}")
        
        
 