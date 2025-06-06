from .base_dataset import *

class AugmentationCaptionDataset(LLaVAInferenceBaseDataset):
    def build_data(self):
        self.input_path = self.data_args.input_path[0]
        self.meta_data = [dict(id=i[:-4],image=os.path.join(self.input_path,i),query="Please describe the image in detail.")  for i in os.listdir(self.input_path)]
        self.resume_from_ckpt()
    
    def resume_from_ckpt(self):
        if os.path.exists(self.data_args.output_path):
            processed_data = process_jsonl(self.data_args.output_path)
            processed_dict = {item["id"]:1 for item in processed_data}
            if len(processed_data) > 0:
                renewed_data = []
                for item in self.meta_data:
                    if processed_dict.get(item["id"],None) is None:
                        renewed_data.append(item)
                self.meta_data = renewed_data
                print_rank0(f"Resumed from checkpoint, remaining data: {len(self.meta_data)}")
        else:
            print_rank0(f"No checkpoint found, start from scratch.")
                   
    # def resume_data_to_file(self,results):
    #     for item in results:
    #         img_id = item["id"]
    #         image_path = f"llava_aug/aug_ft_200k/{img_id}.jpg"
    #         data = {"id":str(img_id),"image":image_path,"caption":item["output"]}
    #         append_jsonl(data,self.data_args.output_path)
    
    def resume_item_to_file(self,item):
        img_id = item["id"]
        target_image_aug_dir = self.data_args.aug_target_dir_path if self.data_args.aug_target_dir_path is not None else "llava_aug/aug_ft_200k"
        image_path = os.path.join(target_image_aug_dir,f"{img_id}.jpg")
        data = {"id":str(img_id),"image":image_path,"caption":item["output"]}
        append_jsonl(data,self.data_args.output_path)
        
class PlainAugmentationCaptionDataset(AugmentationCaptionDataset):
    def build_data(self):
        self.input_path = self.data_args.input_path[0]
        self.image_dir_path = self.data_args.image_dir_path
        raw_data = load_json_file(self.input_path)
        self.meta_data = []
        self.id2img = {}
        for item in raw_data:
            if item.get("image",None) is None:
                continue
            
            id = str(item["id"])
            image = os.path.join(self.image_dir_path,item["image"])
            target_item = dict(id=id,image=image,query="Please describe the image in detail.")
            self.meta_data.append(target_item)
            self.id2img[id] = image
        self.resume_from_ckpt()
    
    def resume_item_to_file(self,item):
        img_id = item["id"]
        data = {"id":str(img_id),"image":self.id2img[img_id],"caption":item["output"]}
        append_jsonl(data,self.data_args.output_path)