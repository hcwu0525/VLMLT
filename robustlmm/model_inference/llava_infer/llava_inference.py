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

from mhr.utils.utils import load_json_file,write_json_file
from mhr.vcd.experiments.eval.language_dict import llava_v1_get_language_conv

# from mhr.alignment.models.llava_v1_5.llava.utils import disable_torch_init
# from mhr.alignment.models.llava_v1_5.llava.model.builder import load_pretrained_model
# from mhr.alignment.models.llava_v1_5.llava.conversation import conv_templates, SeparatorStyle
# from mhr.alignment.models.llava_v1_5.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from mhr.alignment.models.llava_v1_5.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from transformers import HfArgumentParser

# from robustlmm.model_inference.llava_infer.dataset.base_dataset import dataset_dict, DataCollatorForSupervisedDataset
from robustlmm.model_inference.llava_infer.dataset import dataset_dict, DataCollatorForSupervisedDataset
# from .dataset.base_dataset import dataset_dict

def prepare_batch_for_model(batch, device, dtype):
    ids = batch["ids"]
    input_ids = torch.nn.utils.rnn.pad_sequence(batch["input_ids"],
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX).to(device)
    image_tensors = torch.stack(batch["image_tensors"]).to(device, dtype=dtype)
    
    return ids, input_ids, image_tensors


class LLaVAInference():
    
    def __init__(self,inference_args,data_args,model_args):
        
        self.parallel_mode = inference_args.parallel_mode
        self.float_type = inference_args.float_type
        self.inference_args = inference_args
        self.data_args = data_args
        self.data_args.float_type = self.float_type
        self.model_args = model_args
        self.results_lock = threading.Lock()
        
        
            
        if self.parallel_mode == "cpu":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer, self.model, self.image_processor, self.context_len = self.initialize_model(model_args.model_path, device=self.device, peft_model_path=model_args.peft_model_path)
        elif self.parallel_mode == "accelerate":
            self.accelerator = Accelerator()
            self.tokenizer, self.model, self.image_processor, self.context_len = self.initialize_model(model_args.model_path, device=self.accelerator.device, peft_model_path=model_args.peft_model_path)
        else:
            self.distributed_state = PartialState()
            self.device = self.distributed_state.device
            self.tokenizer, self.model, self.image_processor, self.context_len = self.initialize_model(model_args.model_path, device=self.device, peft_model_path=model_args.peft_model_path)
        
        # llava settings
        self.conv_mode = model_args.conv_mode
        self.conv = conv_templates[self.conv_mode].copy()
        self.stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        self.keywords = [self.stop_str] 
        
        # data settings
        dataset_type = self.data_args.dataset_type
        self.dataset = dataset_dict[dataset_type](self.data_args,self.tokenizer,self.image_processor,self.model,self.conv)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.data_args.batch_size, 
                                     collate_fn=DataCollatorForSupervisedDataset(),
                                     shuffle=False,
                                     num_workers=self.data_args.num_workers,
                                     )
        
        
    def prepare_dataset(self):
        pass
            
            
    def initialize_model(self,model_path, device='cuda', peft_model_path=None):
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=model_path, 
                    model_base=None, 
                    model_name=model_name,
                    load_8bit=False, 
                    load_4bit=False, 
                    device=device,
                    device_map=None,
                )
        if peft_model_path:
            model = PeftModel.from_pretrained(model, peft_model_path, adapter_name="dpo")
            print("peft model loaded")
        model.to(self.float_type)
        return tokenizer, model, image_processor, context_len
    
    
    def resume_data_to_file(self):
        self.dataset.resume_data_to_file(self.results)
    
    def inference_on_1gpu(self):
        self.results = []
        self.reslts_id_dict={}
        for batch in tqdm(self.dataloader):
            with self.distributed_state.split_between_processes(batch, apply_padding=True) as _batch:
                
                ids, input_ids, image_tensors = prepare_batch_for_model(_batch, self.device, torch.float16)
                # print("input_ids shape: ",input_ids.shape)
                stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
                # try:
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs=input_ids,
                        images=image_tensors,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=512,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        )

                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                del output_ids, input_ids, image_tensors
                
                # except Exception as e:
                #     print(e)
                #     print(f"error ids: {ids}")
                #     exit(0)
                
                output_item = [dict(id=id, output=output) for id, output in zip(ids, outputs)]
                with self.results_lock:
                    for item in output_item:      
                        if self.reslts_id_dict.get(item["id"],None) is None:
                            item["output"] = item["output"].replace(self.stop_str, '').replace('<unk>', '').replace('<s>', '').replace('</s>', '').strip()
                            self.reslts_id_dict[item["id"]] = True
                            self.results.append(item)
        
        return self.results
    
    def inference_on_accelerate(self):
        self.results = []
        self.reslts_id_dict={}
        
        self.dataloader= self.accelerator.prepare(self.dataloader)
        self.model = self.model.to(self.accelerator.device)
        with torch.inference_mode():
            for batch in tqdm(self.dataloader):
                input_ids, image_tensors = batch["input_ids"], batch["image_tensors"]
                ids = batch["ids"]

                stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)  
                output_ids = self.model.generate(
                    inputs=input_ids.to(self.accelerator.device),
                    images=image_tensors.to(self.accelerator.device),
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )

                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                del output_ids, input_ids, image_tensors
                for id, output in zip(ids, outputs):
                    output_item = dict(id=id, output=output)
                    self.dataset.resume_item_to_file(output_item)
                
        return []
    
    def inference(self):
        if self.parallel_mode == "accelerate":
            return self.inference_on_accelerate()
        else:
            return self.inference_on_1gpu()
    
    
if __name__ == "__main__":

    @dataclass
    class DataArguments:

        input_path: List[str] = field(default_factory=list)
        output_path: str = field(default=None)
        dataset_type: str = field(default="pope_infer")
        batch_size: int = field(default=32)
        coco_path:  Optional[str] = field(default=None)
        gqa_path: Optional[str] = field(default=None)
        image_dir_path: Optional[str] = field(default=None)
        num_workers: int = field(default=4)
        aug_target_dir_path: Optional[str] = field(default=None)
        
    @dataclass
    class ModelArguments:
        model_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b")
        peft_model_path: Optional[str] = field(default=None)
        conv_mode: Optional[str] = field(default="llava_v1")
        
    @dataclass
    class InferenceArguments:
        parallel_mode: str = field(default="1gpu")
        float_type: str = field(default="float16")
        def __post_init__(self):
            self.float_type = getattr(torch, self.float_type)
            
    parser = HfArgumentParser(
    (InferenceArguments, ModelArguments, DataArguments))
    
    
    inference_args, model_args, data_args = parser.parse_args_into_dataclasses()
    inference_module = LLaVAInference(inference_args=inference_args,model_args=model_args,data_args=data_args)
    inference_module.inference()
    inference_module.resume_data_to_file()
    
    