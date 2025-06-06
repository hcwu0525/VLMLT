from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextIteratorStreamer
import sys
from threading import Thread

from tool_server.tf_eval.utils.utils import *
from qwen_vl_utils import process_vision_info

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from thop import profile

image_dir_path="/mnt/petrelfs/share_data/songmingyang/data/mm/imgs"


def generate_conversation_fn(
    text,
    image, 
    role = "user",
):
    messages = [
        {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    ]
    if image:
        messages[0]["content"].append({
                    "type": "image",
                    "image": image
                })

    return messages


def append_conversation_fn(
    conversation, 
    text, 
    image, 
    role
):
    if image:
        new_messages = [
            {
                "role": role,
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    else:
        new_messages = [
            {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    
    conversation.extend(new_messages)

    return conversation

def convert_original_to_qwen_input(item, new_idx=None):
    origin_idx = str(item["id"])
    conversations = item["conversations"]
    image = f"{image_dir_path}/{item['image']}" if "image" in item else None
    for idx,conv in enumerate(conversations):
        conv["value"] = conv["value"].replace("<image>\n", " ")
        conv["value"] = conv["value"].replace("<image>", " ")
        
        if idx == 0:
            assert conv["from"] == "human"
            messages = generate_conversation_fn(conv["value"], image, "user")
        else:
            role = "user" if conv["from"] == "human" else "assistant"
            messages = append_conversation_fn(messages, conv["value"], None, role)
    if new_idx is not None: 
        new_idx = str(new_idx)
        res = dict(id=origin_idx, messages=messages, new_idx=new_idx)
    else:
        res = dict(id=origin_idx, messages=messages)
    return res



class CalculateFLOPsDataset(Dataset):
    def __init__(self,
                 input_path,
                 output_path,
                 processor,
                 image_dir_path="/mnt/petrelfs/share_data/songmingyang/data/mm/imgs"):
        self.input_path = input_path
        self.output_path = output_path
        self.image_dir_path = image_dir_path
        self.processor = processor
        self.load_data()
        self.resume_from_ckpt()
        
    def resume_from_ckpt(self,):
        if os.path.exists(self.output_path):
            print(f"Loading from {self.output_path}")
            ckpt = process_jsonl(self.output_path)
            self.processed_ids = {item["new_idx"]:1 for item in ckpt}
            new_data = []
            
            for item in self.meta_data:
                if item["new_idx"] not in self.processed_ids:
                    new_data.append(item)
            self.meta_data = new_data
            print(f"Resumed from ckpt, remaining {len(new_data)}.")
    
    def load_data(self):
        raw_data = load_json_file(self.input_path)
        self.meta_data = []
        print(f"Loading data from {self.input_path}")
        for idx, item in enumerate(tqdm(raw_data)):
            new_item = convert_original_to_qwen_input(item, new_idx=idx)
            self.meta_data.append(new_item)
    
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        item = self.meta_data[idx]
        messages = item["messages"]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs
        
        res = dict(id=item["id"], inputs=inputs, new_idx=item["new_idx"])
        return res
    
    def save_output(self, item):
        append_jsonl(item, self.output_path)
        



def calculate(args):
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side='left'
    dataset = CalculateFLOPsDataset(args.input_path, args.output_path, processor)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=lambda x: x[0])
    if len(dataloader) == 0:
        print("No data to process.")
        return
    for idx, item in enumerate(tqdm(dataloader)):
        
        try:
            new_idx = item["new_idx"]
            id = item["id"]
            inputs = item["inputs"]
            inputs = inputs.to(model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"] if "pixel_values" in inputs else None
            image_grid_thw = inputs["image_grid_thw"] if "image_grid_thw" in inputs else None
            flops, params = profile(model, inputs=(input_ids, attention_mask, None,
                                        None,None,None,None,None,None,None,pixel_values,None,image_grid_thw))
            res = dict(id=id, new_idx=new_idx, flops=flops, params=params)
            dataset.save_output(res)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA OOM detected, cleaning up...")
                # 清理显存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    args.model_path, torch_dtype="auto", device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(args.model_path)
                processor.tokenizer.padding_side='left'
        # except:
        #     print(f"Error in {idx}")
        #     continue


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, required=True)
    argparser.add_argument("--model_path", type=str, required=True)
    
    args = argparser.parse_args()
    calculate(args)

if __name__ == "__main__":
    main()