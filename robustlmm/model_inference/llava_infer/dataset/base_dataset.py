import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from peft import PeftModel
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass
from typing import Dict, Sequence
from accelerate import PartialState  
from tqdm import tqdm
import os
import torch.distributed as dist

from mhr.utils.utils import load_json_file,write_json_file,write_jsonl,process_jsonl,append_jsonl,print_rank0
from mhr.vcd.experiments.eval.language_dict import llava_v1_get_language_conv

from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

from datasets import load_dataset

class LLaVAInferenceBaseDataset(Dataset):
    def __init__(self,data_args,tokenizer,processor,model,conv) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = processor
        self.data_args = data_args
        self.conv = conv
        self.float_typpe = self.data_args.float_type
        
        self.build_data()

    def build_data(self):
        pass
    
    
    def resume_data_to_file(self,results):
        pass
    
    def img_file_to_pil(self,image):
        if image is None:
            return None
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image
        
    
    def __getitem__(self, index):
        image_path = self.meta_data[index]["image"]
        inp = self.meta_data[index]["query"]
        id = self.meta_data[index]["id"]
        # print(inp)
        
        image = self.img_file_to_pil(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0].to(self.float_typpe)
        
        
        inp = inp.strip().replace('\n', ' ').replace(DEFAULT_IMAGE_TOKEN, '').replace(DEFAULT_IM_START_TOKEN, '').replace(DEFAULT_IM_END_TOKEN, '').replace("<image>","")
        assert DEFAULT_IMAGE_TOKEN not in inp
        assert image is not None
        
        conv = deepcopy(self.conv)
        if image is not None and DEFAULT_IMAGE_TOKEN not in inp:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        assert prompt.count(DEFAULT_IMAGE_TOKEN) == 1, f"prompt: {prompt}, count: {prompt.count(DEFAULT_IMAGE_TOKEN)}"
        assert prompt.count(DEFAULT_IM_START_TOKEN) == 0
        # assert prompt.count("\n") == 0
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        res = {"id":id ,"image_tensor":image_tensor,"input_ids":input_ids}
        return res
    
    def __len__(self):
        return len(self.meta_data)
        
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        ids = [instance["id"] for instance in instances]
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.stack(input_ids)
        image_tensors = [instance["image_tensor"] for instance in instances]
        image_tensors = torch.stack(image_tensors)
        
        batch = dict(
            ids=ids,
            input_ids = input_ids,
            image_tensors = image_tensors,
        )
        return batch
    
