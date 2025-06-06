import os
import torch
import transformers
# import logging
import random


from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
from accelerate import PartialState 
from accelerate.logging import get_logger
from tqdm import tqdm
import torch.distributed as dist

from mhr.utils.utils import *

logger = get_logger(__name__)

class DiffusionGenerationDataset(Dataset):
    def __init__(
        self,
        data_args,
    ):
        self.data_args = data_args
        self.input_path = data_args.input_path
        self.img_dir_path = data_args.img_dir_path
        self.output_path = data_args.output_path
        self.transforms = data_args.transforms
        self.origin_alpha = data_args.origin_alpha
        self.augment_alpha = data_args.augment_alpha
        
        self.raw_data = load_json_file(self.input_path)
        self.build_dr_algo_standards()
        self.inspect_augment_num(self.origin_alpha,self.augment_alpha)
        self.resume_from_ckpt()

    def resume_from_ckpt(self):
        output_img_list = os.listdir(self.output_path)
        output_img_dict = {i:1 for i in output_img_list}
        if len(output_img_list) > 0:
            logger.info("Ckpt detected")
            renewed_data = []
            for item in self.meta_data:
                save_name = str(item['id']).replace("/","_")
                if output_img_dict.get(f"{save_name}_aug.jpg",None) is None:
                    renewed_data.append(item)
            self.meta_data = renewed_data
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"ckpt resumed. len(renewed_data):{len(renewed_data)}")
        else:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"No ckpt detected, skip.")
    
    def __getitem__(self, index) -> Dict:
        item = self.meta_data[index]
        image_path = os.path.join(self.img_dir_path,item["image"])
        image = Image.open(image_path).convert("RGB")
        tsfm_image = self.transforms(image)
        
        return dict(id=item["id"],image=image,transformed_image=tsfm_image)
    
    def __len__(self):
        return len(self.meta_data)
    
    def inspect_augment_num(self,origin_alpha,augment_alpha):
        img_dict={}
        for item in tqdm(self.raw_data):
            if item.get("image",None) is None:
                continue
            if img_dict.get(item["image"],None) is None:
                img_dict[item["image"]] = 1
            else:
                img_dict[item["image"]] = img_dict[item["image"]]+1
                
                
        prob_item_dict={concept:[] for concept in self.compose_list}
        prob_item_dict["max"]=[]
        
        D_star=[]
        D_aug=[]
        for item in tqdm(self.raw_data):
            max_of_prob = 0
            pass_cnt = 0
            for key in self.compose_list:
                for obj in item['statistics'][key]:
                    prob = self.entry_prob[key].get(obj,0)
                    prob_item_dict[key].append({'item':obj,'prob':prob})
                    if random.random() < prob:
                        pass_cnt += 1
                    if prob > max_of_prob:
                        max_of_prob = prob
            if pass_cnt > self.pass_num:
                D_star.append(item)
            if max_of_prob > 1 and item.get("image",None) and img_dict.get(item["image"],1) < 2:
                D_aug.append(item)
                img_dict[item["image"]] = img_dict[item["image"]]+1
            prob_item_dict["max"].append({'item':"-",'prob':max_of_prob})
            
        D_star_new=[]
        for item in D_star:
            if random.random()<origin_alpha:
                D_star_new.append(item)

        D_aug_new=random.sample(D_aug,augment_alpha)
        # for item in D_aug:
        #     if random.random()<augment_alpha:
        #         D_aug_new.append(item)

        for concept,prob_item_list in prob_item_dict.items():
            prob_item_list = sorted(prob_item_list,key=lambda x:x['prob'],reverse=True)
            
        print_rank0(f"len(D_star):{len(D_star)}")
        print_rank0(f"len(D_aug):{len(D_aug)}")
        print_rank0(f"total length:{len(D_star)+len(D_aug)}")
        
        print_rank0(f"len(D_star_new):{len(D_star_new)}")
        print_rank0(f"len(D_aug_new):{len(D_aug_new)}")
        print_rank0(f"total length:{len(D_star_new)+len(D_aug_new)}")

        self.meta_data = D_aug_new
        self.origin_alpha = origin_alpha
        self.augment_alpha = augment_alpha
        
        
            
    def build_dr_algo_standards(self):
        self.pass_num = self.data_args.pass_num
        if self.data_args.threshold_dict is None:
            self.threshold_dict = {'object': 304, 'token': 120, 'co_occurrence': 24, 'what_word': 4895}
            self.reverse_index_file_dict = {
                'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_dino_stat_reverse_index.jsonl',
                'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_token_reverse_index.jsonl',
                'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_co_occurrence_reverse_index.jsonl',
                'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_what_word_reverse_index.jsonl'
            }
        else:
            self.threshold_dict = self.data_args.threshold_dict
            self.reverse_index_file_dict = self.data_args.reverse_index_file_dict
        if self.data_args.compose_list is None:
            self.compose_list = ['object','token','co_occurrence','what_word']
        else:
            self.compose_list = self.data_args.compose_list
        self.entry_prob = self.build_prob_dict(self.reverse_index_file_dict,self.threshold_dict)
        
    def build_prob_dict(self,file_dict,threshold_dict):
        """
        Build a dictionary of probabilities for each entry in the data.
        """
        entry_prob={}
        for key in self.compose_list:
            entry_prob[key] = dict()
            data = process_jsonl(file_dict[key]) 
            for item in data:
                # length = len(item['ids']) if len(item['ids']) > threshold_dict[key] else threshold_dict[key]
                length = len(item['ids']) 
                entry_prob[key][item['object']] = threshold_dict[key] / length 

        return entry_prob
    
    
class ShareGPT4VPTDiffusionGenerationDataset(DiffusionGenerationDataset):
    def __init__(self, data_args):
        data_args.threshold_dict = {'token': 633, 'object': 57, 'co_occurrence': 12, 'what_word': 4895}
        data_args.reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/reverse_index/dinoobj_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/reverse_index/token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/reverse_index/dinocooccur_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_what_word_reverse_index.jsonl'
        }
        data_args.compose_list=['object','token','co_occurrence']
        super().__init__(data_args)
        
    def inspect_augment_num(self,origin_alpha,augment_alpha):
        img_dict={}
        for item in tqdm(self.raw_data):
            if item.get("image",None) is None:
                continue
            if img_dict.get(item["image"],None) is None:
                img_dict[item["image"]] = 1
            else:
                img_dict[item["image"]] = img_dict[item["image"]]+1
                
                
        prob_item_dict={concept:[] for concept in self.compose_list}
        prob_item_dict["max"]=[]
        
        D_star=[]
        D_aug=[]
        for item in tqdm(self.raw_data):
            max_of_prob = 0
            pass_cnt = 0
            for key in self.compose_list:
                # for obj in item['statistics'][key]:
                #     prob = self.entry_prob[key].get(obj,0)
                #     prob_item_dict[key].append({'item':obj,'prob':prob})
                prob_list = [self.entry_prob[key].get(obj,0) for obj in item['statistics'][key]]
                avg_prob = sum(prob_list)/len(prob_list) if len(prob_list) > 0 else 0
                if random.random() < avg_prob:
                    pass_cnt += 1
                if avg_prob > max_of_prob:
                    max_of_prob = avg_prob
            if pass_cnt > self.pass_num:
                D_star.append(item)
            if max_of_prob > 1 and item.get("image",None) and img_dict.get(item["image"],1) < 2:
                D_aug.append(item)
                img_dict[item["image"]] = img_dict[item["image"]]+1
            prob_item_dict["max"].append({'item':"-",'prob':max_of_prob})
            
        D_star_new=[]
        for item in D_star:
            if random.random()<origin_alpha:
                D_star_new.append(item)
        if augment_alpha < len(D_aug):
            D_aug_new=random.sample(D_aug,augment_alpha)
        else:
            D_aug_new = D_aug
        # for item in D_aug:
        #     if random.random()<augment_alpha:
        #         D_aug_new.append(item)

        # for concept,prob_item_list in prob_item_dict.items():
        #     prob_item_list = sorted(prob_item_list,key=lambda x:x['prob'],reverse=True)
            
        print_rank0(f"len(D_star):{len(D_star)}")
        print_rank0(f"len(D_aug):{len(D_aug)}")
        print_rank0(f"total length:{len(D_star)+len(D_aug)}")
        
        print_rank0(f"len(D_star_new):{len(D_star_new)}")
        print_rank0(f"len(D_aug_new):{len(D_aug_new)}")
        print_rank0(f"total length:{len(D_star_new)+len(D_aug_new)}")

        self.meta_data = D_aug_new
        self.origin_alpha = origin_alpha
        self.augment_alpha = augment_alpha
    

class ShareGPT4VFTDiffusionGenerationDataset(DiffusionGenerationDataset):
    def __init__(self, data_args):
        data_args.threshold_dict = {'token': 210, 'object': 68,  'co_occurrence': 10, 'what_word': 203}
        data_args.reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/dinoobj_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/dinocooccur_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/whatword_reverse_index.jsonl'
        }
        data_args.compose_list=['token','object','co_occurrence','what_word']
        super().__init__(data_args)

class LLaVAFTNewDiffusionGenerationDataset(DiffusionGenerationDataset):
    def __init__(self, data_args):
        data_args.threshold_dict = {'object': 211, 'token': 74, 'co_occurrence': 11, 'what_word': 194}
        data_args.reverse_index_file_dict = {
                'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/dinoobj_reverse_index.jsonl',
                'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/token_reverse_index.jsonl',
                'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/dinocooccur_reverse_index.jsonl',
                'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/whatword_reverse_index.jsonl'
            }
        data_args.compose_list=['object','token','co_occurrence','what_word']
        super().__init__(data_args)
        

class LatentDiffusionGenerationDataset():
    def __init__(
        self,
        data_args,
    ):
        self.data_args = data_args
        self.input_path = data_args.input_path
        self.img_dir_path = data_args.img_dir_path
        self.output_path = data_args.output_path
        self.transforms = data_args.transforms
        # self.origin_alpha = data_args.origin_alpha
        # self.augment_alpha = data_args.augment_alpha
        
        self.load_data()
        self.resume_from_ckpt()
    
    def load_data(self):
        raw_data = process_jsonl(self.input_path)
        self.meta_data = raw_data
        

    def resume_from_ckpt(self):
        output_img_list = os.listdir(self.output_path)
        output_img_dict = {i:1 for i in output_img_list}
        if len(output_img_list) > 0:
            logger.info("Ckpt detected")
            renewed_data = []
            for item in self.meta_data:
                save_name = str(item['id']).replace("/","_")
                if output_img_dict.get(f"{save_name}_aug.jpg",None) is None:
                    renewed_data.append(item)
            self.meta_data = renewed_data
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"ckpt resumed. len(renewed_data):{len(renewed_data)}")
        else:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"No ckpt detected, skip.")
    
    
    def __getitem__(self, index) -> Dict:
        item = self.meta_data[index]
        image_path = os.path.join(self.img_dir_path,item["image_path"])
        image = Image.open(image_path).convert("RGB")
        tsfm_image = self.transforms(image)
        
        return dict(id=item["id"],image=image,transformed_image=tsfm_image)
    
    def __len__(self):
        return len(self.meta_data)



@dataclass
class DataCollatorForDiffusionGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        ids = [instance['id'] for instance in instances]
        images = [instance['image'] for instance in instances]
        transformed_images = [instance['transformed_image'] for instance in instances]
        transformed_images = torch.stack(transformed_images)
        
        return dict(ids=ids,images=images,transformed_images=transformed_images)

diffusion_dataset_dict=dict(
    llava_ft=DiffusionGenerationDataset,
    share4v_pt=ShareGPT4VPTDiffusionGenerationDataset,
    share4v_ft=ShareGPT4VFTDiffusionGenerationDataset,
    llava_ft_new=LLaVAFTNewDiffusionGenerationDataset,
    latent_llavaft_self=LatentDiffusionGenerationDataset,
)