import os
import torch
import transformers
import logging


from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from accelerate import PartialState 
from tqdm import tqdm

from mhr.utils.utils import append_jsonl,process_jsonl,load_json_file,print_rank0
from datasets import load_dataset
import inflect

logger = logging.getLogger(__name__)

def str2list(input_str):
    if isinstance(input_str,str):
        raw_list = input_str.strip().replace("\n","").split(",")
        new_list = []
        for item in raw_list:
            new_list.append(item.strip())
        return new_list
    elif isinstance(input_str,list):
        return input_str
    else:
        raise TypeError("input_str should be str or list")
    
class GroundingDinoBaseDataset(Dataset):
    def __init__(
        self,
        data_args,
    ):
        self.data_args = data_args
        self.input_path = data_args.input_path
        self.output_path = data_args.output_path

        self.resume_from_ckpt()
        self.load_data()
    
    def resume_from_ckpt(self):
        assert self.output_path is not None
        if os.path.exists(self.output_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {item["id"]:True for item in self.cache }
        else:
            print_rank0("Output file does not exist, start from scratch")
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def img_file_to_pil(self,image):
        if image is None:
            return None
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image
        else:
            raise TypeError("image should be str or PIL.Image")
        
    
    def load_data(self):
        pass
    
    def __getitem__(self, index) -> Dict:
        item = self.meta_data[index]
        image = self.img_file_to_pil(item["image"])
        image_size = image.size[::-1]
        target_dict = dict(id=item["id"],image=image,image_size=image_size,text=item["text"])
        if "new_idx" in item:
            target_dict["new_idx"] = item["new_idx"]
        return target_dict
    
    def __len__(self):
        return len(self.meta_data)

class POPEGroundingDinoDataset(GroundingDinoBaseDataset):
    def load_data(self):
        self.coco_path = self.data_args.coco_path
        self.gqa_path = self.data_args.gqa_path
        
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for item in raw_data:
            text = item["objects"].replace(",",".")
            image_name = item["id"]
            if image_name.startswith("COCO"):
                image_path = os.path.join(self.coco_path,image_name)
            else:
                image_path = os.path.join(self.gqa_path,image_name)
            self.meta_data.append(dict(id=item["id"],image=image_path,text=text))

class MMEGroundingDinoDataset(GroundingDinoBaseDataset):
    def load_data(self):
        
        self.image_dir_path = self.data_args.image_dir_path 
        
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for item in raw_data:
            text = item["objects"].replace(",",".")
            image_name = item["id"]
            image_path = os.path.join(self.image_dir_path,image_name)
            self.meta_data.append(dict(id=item["id"],image=image_path,text=text))    


class VAL14GroundingDinoDataset(GroundingDinoBaseDataset):
    def resume_from_ckpt(self):
        if os.path.exists(self.output_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {str(item["id"]):True for item in self.cache }
        else:
            print_rank0("Output file does not exist, start from scratch")
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        self.image_dir_path = self.data_args.image_dir_path
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for item in raw_data:
            if self.processed_id.get(str(item["id"]),False) or item.get("image",None) is None:
                continue
            text = ""
            objects = str2list(item["objects"])           
            for idx,token in enumerate(objects):
                if idx < len(objects)-1:
                    text += token + "."
                else:
                    text += token
            image_path = os.path.join(self.image_dir_path,item["image"])
            self.meta_data.append({"id":str(item["id"]),"image":image_path,"text":text})
    
    def write_item_into_file(self,output_item):
        output_item["objects"] = output_item["dino_statistic"]["labels"]
        append_jsonl(output_item,self.output_path)
    
    
class SQAGroundingDinoDataset(GroundingDinoBaseDataset):
    def resume_from_ckpt(self):
        if os.path.exists(self.output_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {str(item["id"]):True for item in self.cache }
        else:
            print_rank0("Output file does not exist, start from scratch")
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        
        raw_data = process_jsonl(self.input_path)
        token_data = process_jsonl(self.data_args.token_path)
        token_dict={item["id"]:item["tokens"] for item in token_data}
        original_data = load_dataset(self.data_args.extra_dataset_suffix,self.data_args.extra_subset_suffix)['test']
        image_dict={str(idx):item["image"] for idx,item in enumerate(original_data)}
        self.meta_data = []
        for item in raw_data:
            id = str(item["id"])
            if self.processed_id.get(id,False) :
                continue
            text = ""
            objects = str2list(item["objects"]) 
            image = image_dict.get(id,None)
            assert image is not None, f"image not found for id: {id}"
            tokens = str2list(token_dict.get(id,[]))
            objects = tokens+objects          
            for idx,token in enumerate(objects):
                if idx < len(objects)-1:
                    text += token + "."
                else:
                    text += token
            self.meta_data.append({"id":id,"image":image,"text":text})
    
    def write_item_into_file(self,output_item):
        target_item = {"id":output_item["id"],"objects":output_item["dino_statistic"]["labels"],"dino_statistic":output_item["dino_statistic"]}
        append_jsonl(target_item,self.output_path)
            
# class ShareGroundingDinoDataset(GroundingDinoBaseDataset):
#     def resume_from_ckpt(self):
#         assert self.output_path is not None
#         if os.path.exists(self.output_path):
#             print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
#             self.cache = process_jsonl(self.output_path)
#             self.processed_id = {item["new_idx"]:True for item in self.cache }
#         else:
#             print_rank0("Output file does not exist, start from scratch")
#             os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
#             self.cache = []
#             self.processed_id = {}
            
#     def load_data(self):
#         self.image_dir_path = self.data_args.image_dir_path
#         self.token_path = self.data_args.token_path
#         raw_data = process_jsonl(self.input_path)
#         token_data = process_jsonl(self.token_path)
#         token_dict={item["new_idx"]:item["tokens"] for item in token_data}
        
#         self.meta_data = []
        
#         for item in raw_data:
#             if self.processed_id.get(item["new_idx"],False) or item.get("image",None) is None:
#                 continue
#             text = ""
#             # tokens = item["statistics"]["token"]+item["statistics"]["object"]
#             tokens = str2list(token_dict.get(item["new_idx"],[]))
#             objects = str2list(item.get("objects",item.get("outputs")))
#             assert objects is not None
#             tokens = tokens+objects
            
#             for idx,token in enumerate(tokens):
#                 if idx < len(tokens)-1:
#                     text += token + "."
#                 else:
#                     text += token
#             image_path = os.path.join(self.image_dir_path,item["image"])
#             self.meta_data.append({"id":item["id"],"image":image_path,"text":text,"new_idx":item["new_idx"]})
    
#     def write_item_into_file(self,output_item):
#         output_item["objects"] = output_item["dino_statistic"]["labels"]
#         append_jsonl(output_item,self.output_path)


# PT is unique id
class SharePTGroundingDinoDataset(GroundingDinoBaseDataset):
    def __init__(self,data_args,):
        self.inflect_engine = inflect.engine()
        super().__init__(data_args)
        
        
        
    def resume_from_ckpt(self):
        assert self.output_path is not None
        if os.path.exists(self.output_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {item["id"]:True for item in self.cache }
        else:
            print_rank0("Output file does not exist, start from scratch")
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        self.image_dir_path = self.data_args.image_dir_path
        self.token_path = self.data_args.token_path
        raw_data = process_jsonl(self.input_path)
        token_data = process_jsonl(self.token_path)
        token_dict={item["id"]:item["tokens"] for item in token_data}
        
        self.meta_data = []
        
        for item in raw_data:
            if self.processed_id.get(item["id"],False) or item.get("image",None) is None:
                continue
            text = ""
            # tokens = item["statistics"]["token"]+item["statistics"]["object"]
            tokens = str2list(token_dict.get(item["id"],[]))
            objects = str2list(item.get("objects",item.get("outputs")))
            assert objects is not None
            tokens = tokens+objects
            tokens = [token.lower() for token in tokens if token != ""]
            tokens = list(set(tokens))
            
            for idx,token in enumerate(tokens):
                if idx < len(tokens)-1:
                    text += token + "."
                else:
                    text += token
            image_path = os.path.join(self.image_dir_path,item["image"])
            self.meta_data.append({"id":item["id"],"image":image_path,"text":text})
        print_rank0(self.meta_data[:10])
    
    def write_item_into_file(self,output_item):
        checked_item = self.filter_item(output_item)
        append_jsonl(checked_item,self.output_path)
    
    def is_plural(self,word):
        # 将单词转换为单数形式并比较
        singular_form =self.inflect_engine.singular_noun(word)
        if singular_form:
            return singular_form
        return None
    
    def filter_item(self,output_item):

        renewed_objects = []
        for obj in output_item["dino_statistic"]["labels"]:
            obj_word_list = obj.strip().lower().split()
            renewd_obj_list = []
            for obj_part in obj_word_list:
                singular_form = self.is_plural(obj_part)
                if singular_form:
                    renewd_obj_list.append(singular_form)
                else:
                    renewd_obj_list.append(obj_part)
            renewd_obj_list = list(set(renewd_obj_list))
            renewd_obj_list = " ".join(renewd_obj_list)
            renewed_objects.append(renewd_obj_list)
        output_item["objects"] = renewed_objects
        return output_item
    
    def filter_list(self,output_list):
    # 创建 inflect 的 engine 实例
        renewed_objects = []
        for obj in output_list:
            obj_word_list = obj.strip().lower().split()
            # print(obj_word_list)
            renewd_obj_list = []
            for obj_part in obj_word_list:
                singular_form = self.is_plural(obj_part)
                if singular_form:
                    renewd_obj_list.append(singular_form)
                else:
                    renewd_obj_list.append(obj_part)
            renewd_obj_list = list(set(renewd_obj_list))
            renewd_obj_list = " ".join(renewd_obj_list)
            renewed_objects.append(renewd_obj_list)
        return renewed_objects

class ShareFTGroundingDinoDataset(SharePTGroundingDinoDataset):
        
    def resume_from_ckpt(self):
        assert self.output_path is not None
        if os.path.exists(self.output_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {item["new_idx"]:True for item in self.cache }
        else:
            print_rank0("Output file does not exist, start from scratch")
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def load_data(self):
        self.image_dir_path = self.data_args.image_dir_path
        self.token_path = self.data_args.token_path
        raw_data = process_jsonl(self.input_path)
        token_data = process_jsonl(self.token_path)
        token_dict={item["new_idx"]:item["tokens"] for item in token_data}
        
        self.meta_data = []
        
        for item in raw_data:
            if self.processed_id.get(item["new_idx"],False) or item.get("image",None) is None:
                continue
            text = ""
            # tokens = item["statistics"]["token"]+item["statistics"]["object"]
            tokens = str2list(token_dict.get(item["new_idx"],[]))
            objects = str2list(item.get("objects",item.get("outputs")))
            assert objects is not None
            tokens = tokens+objects
            tokens = [token.lower() for token in tokens if token != ""]
            # renewed_tokens = self.filter_list(tokens)
            
            tokens = list(set(tokens))
            
            for idx,token in enumerate(tokens):
                if idx < len(tokens)-1:
                    text += token + "."
                else:
                    text += token
            image_path = os.path.join(self.image_dir_path,item["image"])
            self.meta_data.append({"id":item["id"],"image":image_path,"text":text,"new_idx":item["new_idx"]})


# class LLaVATrainsetDinoDataset(ShareFTGroundingDinoDataset):
    
            



@dataclass
class DataCollatorForGroundingDataset(object):
    """Collate examples for supervised fine-tuning."""

    processor: transformers.AutoProcessor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        ids = [instance['id'] for instance in instances]
        text = [instance['text'] for instance in instances]
        images = [instance['image'] for instance in instances]
        image_sizes = [instance['image_size'] for instance in instances]
        
        inputs = self.processor(images=images, text=text, padding='max_length',max_length=256, return_tensors="pt",truncation=True)
        # inputs["ids"] = ids
        # inputs["image_sizes"] = image_sizes
        target_dict = {k:v for k,v in inputs.items()}
        target_dict["pixel_values"] = target_dict["pixel_values"].to(torch.bfloat16)
        target_dict["ids"] = ids
        target_dict["image_sizes"] = image_sizes
        # return dict(ids=ids, image_sizes=image_sizes, inputs=inputs)
        if "new_idx" in instances[0]:
            new_idx = [instance["new_idx"] for instance in instances]
            target_dict["new_idx"] = new_idx
        return target_dict

dataset_dict = {
    "base":GroundingDinoBaseDataset,
    "pope_infer":POPEGroundingDinoDataset,
    "mme_infer":MMEGroundingDinoDataset,
    "sharegpt4v_infer":ShareFTGroundingDinoDataset,
    "val14_infer":VAL14GroundingDinoDataset,
    "sqa_infer":SQAGroundingDinoDataset,
    "sharegpt4v_pt_infer":SharePTGroundingDinoDataset
}