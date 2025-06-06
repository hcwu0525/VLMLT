import os
import torch
import transformers
# import logging
import random


from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union
from PIL import Image
import numpy as np

from mhr.utils.utils import *
import sys
sys.path.append('/mnt/petrelfs/songmingyang/code/mm/robustLMM/ref/ControlNet')
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from diffusers.utils import load_image, make_image_grid
# apply_hed = HEDdetector()


class ControlnetGenerationDataset():
    def __init__(
        self,
        data_args,
    ):
        self.data_args = data_args
        self.input_path = data_args.input_path
        self.img_dir_path = data_args.img_dir_path
        self.output_path = data_args.output_path
        self.detect_resolution = data_args.detect_resolution
        self.image_resolution = data_args.image_resolution
        
        self.load_data()
        self.resume_from_ckpt()
    
    def load_data(self):
        raw_data = process_jsonl(self.input_path)
        self.meta_data = raw_data
        

    def resume_from_ckpt(self):
        output_img_list = os.listdir(self.output_path)
        output_img_dict = {i:1 for i in output_img_list}
        if len(output_img_list) > 0:
            renewed_data = []
            for item in self.meta_data:
                save_name = str(item['id']).replace("/","_")
                if output_img_dict.get(f"{save_name}_aug.jpg",None) is None:
                    renewed_data.append(item)
            self.meta_data = renewed_data
            print_rank0(f"ckpt resumed. len(renewed_data):{len(renewed_data)}")
        else:
            print_rank0(f"No ckpt detected, skip.")
    
    
    def __getitem__(self, index) -> Dict:
        item = self.meta_data[index]
        image_path = os.path.join(self.img_dir_path,item["image"])
        image = np.array(load_image(image_path))
        img = resize_image(image, self.image_resolution)
        
        return dict(id=item["id"],image=img)
    
    def __len__(self):
        return len(self.meta_data)
    
    def write_item(self,item):
        id,results = item["id"],item["results"]
        save_name = str(id).replace("/","_")
        save_name = os.path.join(self.output_path,f"{save_name}_aug.jpg")
        out_img=Image.fromarray(results[0])
        out_img.save(save_name)



@dataclass
class DataCollatorForDiffusionGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        ids = [instance['id'] for instance in instances]
        images = [instance['image'] for instance in instances]
        
        return dict(ids=ids,images=images)

diffusion_dataset_dict=dict(
    base=ControlnetGenerationDataset,
)