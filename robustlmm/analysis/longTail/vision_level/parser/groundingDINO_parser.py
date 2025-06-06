import os
import torch
import transformers
import logging


from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
from accelerate import PartialState 
from tqdm import tqdm

from mhr.utils.utils import append_jsonl,process_jsonl

logger = logging.getLogger(__name__)

class GroundingDataset(Dataset):
    def __init__(
        self,
        input_file_path: str = None,
        output_file_path: str = None,
        image_dir_path: str = None,
    ):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.image_dir_path = image_dir_path
        
        self.resume_from_ckpt()
        self.load_data()
    
    def resume_from_ckpt(self):
        assert self.output_file_path is not None
        if os.path.exists(self.output_file_path):
            logger.info(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {item["id"]:True for item in self.cache }
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    
    def load_data(self):
        assert self.input_file_path is not None
        raw_data = process_jsonl(self.input_file_path)
        self.data=[]
        for item in raw_data:
            if self.processed_id.get(item["id"],False):
                continue
            if item.get("image",None) is None: 
                continue
            data_id = item["id"]
            image_path = os.path.join(self.image_dir_path,item["image"])
            res=""
            for word in list(set(item["objects"]["pos"]+item["objects"]["llama"])):
                res += word+"."
            self.data.append(dict(id=data_id,image=image_path,text=res))
        logger.info(f"Data loaded from {self.input_file_path}, total {len(self.data)} items.")

        
    
    def __getitem__(self, index) -> Dict:
        item = self.data[index]
        image = Image.open(item["image"]).convert("RGB")
        image_size = image.size[::-1]
        
        return dict(id=item["id"],image=image,image_size=image_size,text=item["text"],)
    
    def __len__(self) -> int:
        return len(self.data)


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
        inputs["ids"] = ids
        inputs["image_sizes"] = image_sizes
        # return dict(ids=ids, image_sizes=image_sizes, inputs=inputs)
        return inputs


class BaseGroudingDINOParser():
    
    def __init__(
            self,
            model_path: str = "/mnt/petrelfs/songmingyang/songmingyang/model/mm/grounding-dino-base",
            input_file_path: str = None,
            image_dir_path: str = None,
            output_file_path: str = None,
            batch_size: int = 32,
        ) -> None:
        
        
        self.distributed_state = PartialState()
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path,device_map=self.distributed_state.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.dataset = GroundingDataset(input_file_path=input_file_path, output_file_path=output_file_path,image_dir_path=image_dir_path)
        self.data_collator = DataCollatorForGroundingDataset(self.processor)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.data_collator, num_workers=2, shuffle=False)
        self.model.eval()
        
    def inference(self):
        pass
    
    def parallel_process(self):
        
        for batch in tqdm(self.dataloader):
            with self.distributed_state.split_between_processes(batch) as input_dict:
                # input_dict = input_dict.to(self.distributed_state.device)
                ids = input_dict.pop("ids")
                image_sizes = input_dict.pop("image_sizes")
                inputs = {k: v.to(self.distributed_state.device) for k, v in input_dict.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                generated_text = self.processor.post_process_grounded_object_detection(
                    outputs,
                    input_dict["input_ids"],
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=image_sizes
                )          
                assert len(ids) == len(generated_text)
                for i in range(len(ids)):
                    item = dict(id=ids[i],statistic=dict(labels=generated_text[i]["labels"],boxes=generated_text[i]["boxes"].cpu().tolist()))
                    append_jsonl(item,self.dataset.output_file_path)
                del outputs,generated_text,inputs,ids,image_sizes,input_dict
                
    
    