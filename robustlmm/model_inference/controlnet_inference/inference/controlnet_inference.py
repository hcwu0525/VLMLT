import sys
sys.path.append('/mnt/petrelfs/songmingyang/code/mm/robustLMM/ref/ControlNet')
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
from PIL import Image
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from diffusers.utils import load_image, make_image_grid
from accelerate import Accelerator
from torch.utils.data import DataLoader,Dataset
from transformers import HfArgumentParser
from annotator.hed import HEDdetector_accelerator
from diffusers.utils import load_image, make_image_grid



from controlnet_inference_dataset import ControlnetGenerationDataset,DataCollatorForDiffusionGenerationDataset

class ControlnetInference():
    def __init__(self,data_args,inference_args) -> None:
        self.data_args = data_args
        self.inference_args = inference_args
        self.model_config_path = self.inference_args.model_config_path
        self.model_ckpt_path = self.inference_args.model_ckpt_path
        self.detect_resolution = data_args.detect_resolution
        self.image_resolution = data_args.image_resolution
        
        self.accelerator = Accelerator()
        
        self.hyper_dict = dict(
                    input_image=None,
                    prompt="default",
                    a_prompt="best quality, extremely detailed",
                    n_prompt="longbody, lowres, bad anatomy, bad hands, bad faces, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, art, cartoon",
                    num_samples=1,
                    image_resolution=data_args.image_resolution,
                    detect_resolution=data_args.detect_resolution,
                    ddim_steps=20,
                    guess_mode=False,
                    strength=1.0,
                    scale=9.0,
                    seed=inference_args.seed,
                    eta=0.0,
                )
        
        self.init_model()
        self.init_data()
    
    def init_data(self):
        self.dataset = ControlnetGenerationDataset(self.data_args)
        self.dataloader = DataLoader(self.dataset,
                                     collate_fn=DataCollatorForDiffusionGenerationDataset(),
                                     batch_size=self.data_args.batch_size,
                                     num_workers=self.data_args.num_workers,)
        assert self.data_args.batch_size == 1, "batch_size must be 1"
        
        
    
    def init_model(self):
        model = create_model(self.model_config_path).cpu()
        model.load_state_dict(load_state_dict(self.model_ckpt_path, location='cpu'))
        self.model = model
        
        self.model.eval()
        self.model = self.model.to(self.accelerator.device)
        self.ddim_sampler = DDIMSampler(self.model)
        
        self.apply_hed = HEDdetector_accelerator(self.accelerator)
        
    
    def process(self):
        
        
        self.dataloader = self.accelerator.prepare(self.dataloader)
        
        seed = self.hyper_dict["seed"]
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        with torch.no_grad():    
            for batch in tqdm(self.dataloader):
                image = batch["images"][0]
                id = batch["ids"][0]
                
                try:
                    detected_map = self.apply_hed(resize_image(image, self.detect_resolution))
                    detected_map = HWC3(detected_map)
                    
                    H, W, C = image.shape
                    input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta \
                    = image, self.hyper_dict["prompt"], self.hyper_dict["a_prompt"], self.hyper_dict["n_prompt"], self.hyper_dict["num_samples"], self.hyper_dict["image_resolution"], self.hyper_dict["detect_resolution"], self.hyper_dict["ddim_steps"], self.hyper_dict["guess_mode"], self.hyper_dict["strength"], self.hyper_dict["scale"], self.hyper_dict["seed"], self.hyper_dict["eta"]

                    
                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(num_samples)], dim=0)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

                    

                    if config.save_memory:
                        self.model.low_vram_shift(is_diffusing=False)

                    cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
                    shape = (4, H // 8, W // 8)
                    
                    if config.save_memory:
                        self.model.low_vram_shift(is_diffusing=True)

                    self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                    samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                                shape, cond, verbose=False, eta=eta,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=un_cond)

                    if config.save_memory:
                        self.model.low_vram_shift(is_diffusing=False)

                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

                    results = [x_samples[i] for i in range(num_samples)]
                    self.dataset.write_item({"id":id,"results":results})
                except Exception as e:
                    print(f"Error in {id}")
                    print(e)
                    continue
                
                
                
if __name__ == "__main__":
    @dataclass
    class DataArguments:
        input_path: str = field(default_factory=str)
        img_dir_path: str = field(default=None)
        output_path: str = field(default=None)
        batch_size: int = field(default=1)
        num_workers: int = field(default=2)
        detect_resolution: int = field(default=512)
        image_resolution: int = field(default=512)
        
        
        
    @dataclass
    class InferenceArguments:
        model_config_path: str = field(default="/mnt/petrelfs/songmingyang/code/mm/robustLMM/ref/ControlNet/models/cldm_v15.yaml")
        model_ckpt_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/mm/diffusers/ControlNet/models/control_sd15_hed.pth")
        seed: int = field(default=42)
 
    
    parser = HfArgumentParser(
    (InferenceArguments, DataArguments))
    
    
    inference_args, data_args = parser.parse_args_into_dataclasses()
    random.seed(inference_args.seed)
    model = ControlnetInference(data_args, inference_args)
    model.process()

        
    

