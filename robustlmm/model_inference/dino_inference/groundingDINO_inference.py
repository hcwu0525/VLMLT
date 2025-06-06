import os
import torch
import transformers
import logging


from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor,HfArgumentParser
from accelerate import PartialState,Accelerator
from tqdm import tqdm
from typing import List,Dict,Optional

from mhr.utils.utils import append_jsonl,process_jsonl
from robustlmm.model_inference.dino_inference.dino_dataset import dataset_dict,DataCollatorForGroundingDataset

logger = logging.getLogger(__name__)



class BaseGroudingDINOParser():
    
    def __init__(
            self,
            model_args = None,
            data_args = None,
            inference_args = None,
        ) -> None:
        
        self.model_args = model_args
        self.inference_args = inference_args
        self.data_args = data_args
        
        # self.distributed_state = PartialState()
        self.accelerator = Accelerator()
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_args.model_path).to(self.inference_args.float_type)
        self.processor = AutoProcessor.from_pretrained(model_args.model_path)
        
        self.dataset = dataset_dict[data_args.dataset_type](data_args)
        self.data_collator = DataCollatorForGroundingDataset(self.processor)
        self.dataloader = DataLoader(self.dataset, batch_size=data_args.batch_size, collate_fn=self.data_collator, num_workers=2, shuffle=False)
        self.model.eval()
        
        
    def inference(self):
        pass
    
    def parallel_process(self):
        self.model,self.dataloader = self.accelerator.prepare(self.model,self.dataloader)
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                ids = batch.pop("ids")
                image_sizes = batch.pop("image_sizes")
                if "new_idx" in batch:
                    new_idxs = batch.pop("new_idx")
                else:
                    new_idxs = None
                inputs = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                # inputs["pixel_values"] = inputs["pixel_values"].to(self.inference_args.float_type)
                outputs = self.model(**inputs)
                generated_text = self.processor.post_process_grounded_object_detection(
                    outputs,
                    batch["input_ids"],
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=image_sizes
                )          
                assert len(ids) == len(generated_text)
                for i in range(len(ids)):
                    item = dict(id=ids[i],dino_statistic=dict(labels=generated_text[i]["labels"],boxes=generated_text[i]["boxes"].cpu().tolist()))
                    if new_idxs is not None:
                        item["new_idx"] = new_idxs[i]
                    # append_jsonl(item,self.dataset.output_path)
                    self.dataset.write_item_into_file(item)
                del outputs,generated_text,inputs,ids,image_sizes
                
if __name__ == "__main__":
    @dataclass
    class DataArguments:

        input_path: str = field(default=None)
        token_path: str = field(default=None)
        output_path: str = field(default=None)
        dataset_type: str = field(default="pope_infer")
        batch_size: int = field(default=32)
        coco_path:  Optional[str] = field(default=None)
        gqa_path: Optional[str] = field(default=None)
        image_dir_path: Optional[str] = field(default=None)
        extra_subset_suffix: Optional[str] = field(default=None)
        extra_dataset_suffix: Optional[str] = field(default=None)
        
    @dataclass
    class ModelArguments:
        model_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/mm/grounding-dino-base")
    
        
    @dataclass
    class InferenceArguments:
        parallel_mode: str = field(default="1gpu")
        float_type: str = field(default="bfloat16")
        def __post_init__(self):
            self.float_type = getattr(torch, self.float_type)
            
    parser = HfArgumentParser(
    (InferenceArguments, ModelArguments, DataArguments))
    
    
    inference_args, model_args, data_args = parser.parse_args_into_dataclasses()
    inference_module = BaseGroudingDINOParser(inference_args=inference_args,model_args=model_args,data_args=data_args)
    inference_module.parallel_process()
    