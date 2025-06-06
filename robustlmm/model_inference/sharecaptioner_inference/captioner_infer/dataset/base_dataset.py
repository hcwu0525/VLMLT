import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import transformers
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
from copy import deepcopy

class ShareCaptionerDataset(Dataset):
    def __init__(self, args):
       self.args = args
       self.model = args.model
       self.input_path = args.input_path[0]
       self.output_path = args.output_path
       self.image_rel_path = args.image_rel_path
       self.vis_processor = deepcopy(self.model.vis_processor)
       self.load_data()
    
    def load_data(self):
        raw_data = os.listdir(self.input_path)
        self.metadata = []
        for image in raw_data:
            image_path = os.path.join(self.input_path, image)
            image_id = image.split(".")[0]
            self.metadata.append({"id": image_id, "image": image_path})
        print_rank0(f"Load Data: {len(self.metadata)}")
        self.resume_from_ckpt()
    
    def resume_from_ckpt(self):
        if os.path.exists(self.output_path):
            processed_data = process_jsonl(self.output_path)
            processed_dict = {item["id"]:1 for item in processed_data}
            if len(processed_data) > 0:
                renewed_data = []
                for item in self.metadata:
                    if processed_dict.get(item["id"],None) is None:
                        renewed_data.append(item)
                self.metadata = renewed_data
                print_rank0(f"Resumed from checkpoint, remaining data: {len(self.metadata)}")
    
    def __getitem__(self, index):
        item = self.metadata[index]
        image_embed = self.process_image(item["image"])
        return {"id": item["id"], "image": image_embed}
    
    def __len__(self):
        return len(self.metadata)
    
    def process_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        img_embed = self.vis_processor(image)
        # img_embed = self.model.encode_img(img_embed)
        return img_embed
    
    def write_output_item(self, item):
        item["image"] = f"{self.image_rel_path}/{item['id']}.jpg"
        append_jsonl(item, self.output_path)
    
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        ids = [instance["id"] for instance in instances]
        ## text input
        # seg1 = '<|User|>:'
        # seg2 = f'Analyze the image in a comprehensive and detailed manner.{self.model.eoh}\n<|Bot|>:'
        # seg_emb1 = self.model.encode_text(seg1, add_special_tokens=True)
        # seg_emb2 = self.model.encode_text(seg2, add_special_tokens=False)
        embeddings = [instance["image"] for instance in instances]
        embeddings = torch.stack(embeddings, dim=0)
        batch = dict(
            ids=ids,
            img_embed = embeddings,
        )
        return batch