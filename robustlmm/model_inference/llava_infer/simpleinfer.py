import os
import json
import tqdm
import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
# from peft import PeftModel
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass, field

from typing import Dict, Sequence, Optional,List
# from accelerate import PartialState,Accelerator
from tqdm import tqdm
from functools import partial
import threading

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

def initialize_model(model_path, device='cuda', peft_model_path=None):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path, 
                model_base=None, 
                model_name=model_name,
                load_8bit=False, 
                load_4bit=False, 
                device=device,
            )
    if peft_model_path:
        model = PeftModel.from_pretrained(model, peft_model_path, adapter_name="dpo")
        print("peft model loaded")
    model.to(torch.float16)
    return tokenizer, model, image_processor, context_len


image_path = "/mnt/petrelfs/songmingyang/code/mm/robustLMM/robustlmm/model_inference/llava_infer/samples/test1.jpg"
inp = "Please Describe this image in detail"
# print(inp)
model_path = "/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = initialize_model(model_path, device="cuda")

image = Image.open(image_path).convert("RGB")
image_tensor = process_images([image], image_processor, model.config).to(model.dtype).to(model.device)

conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()
inp = inp.strip().replace('\n', ' ').replace(DEFAULT_IMAGE_TOKEN, '').replace(DEFAULT_IM_START_TOKEN, '').replace(DEFAULT_IM_END_TOKEN, '').replace("<image>","")
assert DEFAULT_IMAGE_TOKEN not in inp
assert image is not None

if image is not None and DEFAULT_IMAGE_TOKEN not in inp:
    # first message
    if model.config.mm_use_im_start_end:
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
assert prompt.count(DEFAULT_IMAGE_TOKEN) == 1
assert prompt.count(DEFAULT_IM_START_TOKEN) == 0
# assert prompt.count("\n") == 0

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)


generation_num=1
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]   
    


stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        inputs=input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        )
    outputs = tokenizer.batch_decode(output_ids[: , input_ids.shape[1]:])        


print(outputs)