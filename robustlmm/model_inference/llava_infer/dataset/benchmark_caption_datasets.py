from .base_dataset import *

class POPEInferenceDataset(LLaVAInferenceBaseDataset):
    
    def build_data(self):
        if not isinstance(self.data_args.input_path, list):
            self.input_path = [self.data_args.input_path]
        else:
            self.input_path = self.data_args.input_path
        
        self.meta_data_origin = []
        for file in self.input_path:
            data_list = process_jsonl(file)
            self.meta_data_origin.extend(data_list)
            
        self.img_dict = {}
        for item in self.meta_data_origin:
            image_file_name = item.get("image", None)
            if image_file_name is None:
                continue
            if image_file_name.startswith("COCO"):
                image_path = os.path.join(self.data_args.coco_path,image_file_name)
            else:
                image_path = os.path.join(self.data_args.gqa_path,image_file_name)
            self.img_dict[image_file_name] = image_path
        self.meta_data = [dict(id=k,image=v,query="Please describe the image in detail.") for k,v in self.img_dict.items()]

    def resume_data_to_file(self,results):
        write_jsonl(results,self.data_args.output_path)
        

class MMEInferenceDataset(LLaVAInferenceBaseDataset):
    
    def build_data(self):
        self.input_path = self.data_args.input_path[0]
        types = [i for i in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path,i)) and i != "eval_tool"]
        anno_dict = {}
        for type_item in types:
            annotation_path = os.path.join(self.input_path,type_item,"json_labels",f"mme_{type_item}_en.json")
            annotation_data = process_jsonl(annotation_path)
            for item in annotation_data:
                anno_dict[item["image"]] = os.path.join(self.input_path,item["image"])
                
        self.meta_data = [dict(id=k,image=v,query="Please describe the image in detail.") for k,v in anno_dict.items()]
   
    def resume_data_to_file(self,results):
        for item in results:
            image_file_name = item["id"]
            type_name = image_file_name.strip().split("/")[0]
            append_jsonl(item,os.path.join(self.data_args.output_path,f"mme_{type_name}_captions.json"))


class COCO14VALDataset(LLaVAInferenceBaseDataset):
    def build_data(self):
        self.input_path = self.data_args.input_path[0]
        self.image_dir_path = self.data_args.image_dir_path
        self.meta_data = []
        raw_data = load_json_file(self.input_path)["images"]
    
        self.id2filename = {str(item["id"]):item["file_name"] for item in raw_data}
        for item in raw_data:
            target_meta = dict(id=str(item["id"]),image=os.path.join(self.image_dir_path,item["file_name"]),query="Please describe the image in detail.")
            self.meta_data.append(target_meta)
        self.resume_from_ckpt()
        
    def resume_from_ckpt(self):
        if os.path.exists(self.data_args.output_path):
            processed_data = process_jsonl(self.data_args.output_path)
            processed_dict = {str(item["id"]):1 for item in processed_data}
            if len(processed_data) > 0:
                renewed_data = []
                for item in self.meta_data:
                    if processed_dict.get(item["id"],None) is None:
                        renewed_data.append(item)
                self.meta_data = renewed_data
                print_rank0(f"Resumed from checkpoint, remaining data: {len(self.meta_data)}")
        else:
            print_rank0(f"No checkpoint found, start from scratch.")
            
    def resume_item_to_file(self,item):
        img_id = item["id"]
        image_path = self.id2filename[img_id]
        data = {"id":str(img_id),"image":image_path,"caption":item["output"]}
        append_jsonl(data,self.data_args.output_path)

class SQACaptionDataset(COCO14VALDataset):
    def build_data(self):
        self.input_path = self.data_args.input_path[0]
        raw_data = load_dataset(self.data_args.input_path[0],self.data_args.input_path[1])['test']
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            id = str(idx)
            image=item["image"]
            if image is None:
                continue
            target_item = dict(id=id,image=image,query="Please describe the image in detail.")
            self.meta_data.append(target_item)
        self.resume_from_ckpt()
            
    def resume_item_to_file(self,item):
        target_item={"id":item["id"],"caption":item["output"]}
        append_jsonl(target_item,self.data_args.output_path)