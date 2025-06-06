import os
import json
import tqdm
import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from peft import PeftModel
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass, field

from typing import Dict, Sequence, Optional,List
from accelerate import PartialState,Accelerator
from tqdm import tqdm
from functools import partial
import threading
from tqdm import tqdm

from mhr.utils.utils import *
from mhr.vcd.experiments.eval.language_dict import llava_v1_get_language_conv

from .dataset import dataset_dict,DataCollatorForSupervisedDataset
from transformers import AutoTokenizer,AutoModel
def get_rid_of_substr(substr,list_of_target_str):
    renewed_target = []
    for temp_str in list_of_target_str:
        for temp_sub in substr:
            temp_str = temp_str.replace(temp_sub,"")
        renewed_target.append(temp_str)
    return renewed_target

special_tokens = ["<s>","</s>","<TOKENS_UNUSED_1>","<unk>"]
class ShareCaptionerInference():
    
    def __init__(self,inference_args,data_args):
        self.args = inference_args
        self.data_args = data_args
        self.model_path = inference_args.model_path
        self.dtype = getattr(torch, inference_args.float_type)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True,device="cpu")
        self.model.tokenizer = self.tokenizer
        self.model=self.model.to(self.dtype)
        
        self.accelerator = Accelerator()
        data_args.model = self.model
        
        self.dataset = dataset_dict[data_args.dataset_type](data_args)
        self.dataloader = DataLoader(self.dataset, batch_size=data_args.batch_size, num_workers=data_args.num_workers,collate_fn=DataCollatorForSupervisedDataset())
        
    
    def inference(self):
        self.model = self.model.to(self.accelerator.device)
        self.dataloader = self.accelerator.prepare(self.dataloader)
        batch_size = self.data_args.batch_size
        seg1 = '<|User|>:'
        seg2 = f'Analyze the image in a comprehensive and detailed manner.{self.model.eoh}\n<|Bot|>:'
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                ids = batch["ids"]
                img_embed = batch["img_embed"].to(self.accelerator.device).to(self.model.dtype)
                img_embed = self.model.encode_img(img_embed)
                seg_emb1 = self.model.encode_text(seg1, add_special_tokens=True)
                seg_emb2 = self.model.encode_text(seg2, add_special_tokens=False)
                tmp_seg_emb1 = seg_emb1.repeat(batch_size, 1, 1).to(self.accelerator.device).to(self.model.dtype)
                tmp_seg_emb2 = seg_emb2.repeat(batch_size, 1, 1).to(self.accelerator.device).to(self.model.dtype)
                input_emb = torch.cat([tmp_seg_emb1, img_embed, tmp_seg_emb2], dim=1)
                out_embeds = self.model.internlm_model.generate(inputs_embeds=input_emb,
                                                           max_length=500,
                                                           num_beams=3,
                                                           min_length=1,
                                                           do_sample=True,
                                                           repetition_penalty=1.5,
                                                           length_penalty=1.0,
                                                           temperature=1.,
                                                           eos_token_id=self.model.tokenizer.eos_token_id,
                                                           num_return_sequences=1,
                                                           )

                out_embeds[out_embeds == -1] = 2
                out_text = self.tokenizer.batch_decode(out_embeds, skip_special_tokens=True)
                out_text = get_rid_of_substr(special_tokens,out_text)
                # del out_embeds, img_embed, input_emb, tmp_seg_emb1, tmp_seg_emb2,seg_emb1,seg_emb2
                # torch.cuda.empty_cache()
                for idx, text in zip(ids, out_text):
                    self.dataset.write_output_item(dict(id=idx, caption=text))
                

    

    