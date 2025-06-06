import os
import tqdm
import torch
import random
import logging

from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm
from accelerate import Accelerator
from transformers import HfArgumentParser,AutoProcessor,AutoModelForZeroShotImageClassification
from diffusers import StableDiffusionImageVariationPipeline
from diffusion_dataset import DiffusionGenerationDataset,DataCollatorForDiffusionGenerationDataset,diffusion_dataset_dict

from accelerate.logging import get_logger
logger = get_logger(__name__)
class DiffusionGenerationModel():
    def __init__(self,
                 data_args,
                    model_args,
                    inference_args):
        self.data_args = data_args
        self.model_args = model_args
        self.inference_args = inference_args
        
        self.candidate_img_num = self.inference_args.candidate_img_num
        self.output_path = self.data_args.output_path
        # self.output_log = self.data_args.output_log
        self.function = self.inference_args.function
        
        self.init_model()
        self.data_args.transforms = self.diffusion_transform
        self.init_data_module()
        
    
    
    def init_data_module(self):
        # self.dataset = DiffusionGenerationDataset(self.data_args)
        self.dataset = diffusion_dataset_dict[self.function](self.data_args)
        self.dataloader = DataLoader(self.dataset,
                                     collate_fn=DataCollatorForDiffusionGenerationDataset(),
                                     batch_size=self.data_args.batch_size,
                                     num_workers=self.data_args.num_workers,)
    
    def init_model(self) -> None:
        diffusion_model_name_or_path = self.model_args.diffusion_model_name_or_path
        clip_model_name_or_path = self.model_args.clip_model_name_or_path
        
        self.diffusion_model = StableDiffusionImageVariationPipeline.from_pretrained(diffusion_model_name_or_path, revision = "v2.0")
        self.diffusion_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
                ),
            transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
        ])
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
        self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained(clip_model_name_or_path)
        
        if self.inference_args.parallel_mode == "1gpu":
            self.device = self.model_args.device
            self.diffusion_model = self.diffusion_model.to(self.device)
            self.clip_model = self.clip_model.to(self.device)
        elif self.inference_args.parallel_mode == "accelerate":
            self.accelerator = Accelerator()
            
        else:
            raise ValueError("Invalid parallel mode")
    
    def inference(self):
        self.dataloader, self.diffusion_model, = self.accelerator.prepare(self.dataloader, self.diffusion_model,)
        self.clip_model = self.clip_model.to(self.accelerator.device)
        # self.diffusion_model.safety_checker = lambda images, clip_input: (images, False)
        self.diffusion_model = self.diffusion_model.to(self.accelerator.device)
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                # print(f"batch size: {len(batch['images'])}")
                candidates = [ [x] for x in batch["images"] ]
                for sample_time in range(self.candidate_img_num):
                    gen_images = self.diffusion_model(
                        batch["transformed_images"].to(self.accelerator.device),
                        guidance_scale = 3,
                    ).images
                    for i in range(len(gen_images)):
                        candidates[i].append(gen_images[i])
                for idx,sample in enumerate(candidates):
                    inputs = self.clip_processor(images=sample,return_tensors="pt").to(self.accelerator.device)
                    logits = self.clip_model.get_image_features(**inputs)
                    rank = self.select_confident_samples_cosine(logits)
                    save_name = str(batch['ids'][idx]).replace("/","_")
                    sample[rank[0]].save(os.path.join(self.output_path, f"{save_name}_aug.jpg"))
                
    def select_confident_samples_cosine(self,logits,):
        cosine_distan = [torch.nn.CosineSimilarity(dim=0)(logits[0], logits[i]) for i in range(1, logits.shape[0])]
        cosine_distan = torch.stack(cosine_distan)
        idx_cosine = torch.argsort(cosine_distan, descending=True)[:int(cosine_distan.size()[0])]
        # idx
        for i in range(idx_cosine.shape[0]):
            idx_cosine[i] +=1
        logits_cos = logits[idx_cosine]

        return idx_cosine    

if __name__ == "__main__":
    @dataclass
    class DataArguments:

        input_path: str = field(default_factory=str)
        img_dir_path: str = field(default=None)
        output_path: str = field(default=None)
        # output_log: str = field(default=None)
        dataset_type: str = field(default="pope_infer")
        batch_size: int = field(default=64)
        num_workers: int = field(default=4)
        origin_alpha: float = field(default=0.8)
        augment_alpha: int = field(default=100000)
        pass_num: float = field(default=0)
        threshold_dict: dict = field(default=None)
        compose_list: dict = field(default=None)
        reverse_index_file_dict: dict = field(default=None)
        
    @dataclass
    class ModelArguments:
        diffusion_model_name_or_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/mm/sd-image-variations-diffusers")
        clip_model_name_or_path:str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/others/clip-vit-large-patch14-336")
        device: str = field(default="cuda")
        
    @dataclass
    class InferenceArguments:
        parallel_mode: str = field(default="accelerate")
        float_type: str = field(default="float16")
        candidate_img_num: int = field(default=5)
        function: str=field(default="llava_ft")
        seed: int = field(default=42)
        def __post_init__(self):
            self.float_type = getattr(torch, self.float_type)
    
    parser = HfArgumentParser(
    (InferenceArguments, ModelArguments, DataArguments))
    
    
    inference_args, model_args, data_args = parser.parse_args_into_dataclasses()
    random.seed(inference_args.seed)
    model = DiffusionGenerationModel(data_args, model_args, inference_args)
    model.inference()
