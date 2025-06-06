from .base_inference import *

class ShareGPT4VTrainsetParser(LlamaParser):
    def resume_from_file(self):
        if os.path.exists(self.output_file_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {item["new_idx"]:1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def prepare_dataset(self):
        raw_data = load_json_file(self.input_file_path)
        messages = self.prepare_prompt()
        
        # self.data_dict = {i['id']:i for i in raw_data}
        self.meta_data = []
        logger.info(f"Start loading samples")
        for idx,item in enumerate(tqdm(raw_data)):
            if self.processed_id.get(str(idx),False):
                continue
            image = item.get("image","None")
            messages_iter = deepcopy(messages)
            res_str = ""
            for conv_idx,conv in enumerate(item["conversations"]):
                res_str += f"{conv['from']}: {conv['value']}\n"
                
            messages_iter.append({"role": "user", "content": res_str})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":str(item['id']),"prompt":prompt,"image":image,"new_idx":str(idx)})
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
    
    def post_process_batch(self, batch):
        generated_text=batch["generated_text"]
        ids = batch["id"]
        images = batch.get("image",[None]*len(ids))
        new_idx = batch["new_idx"]
        for i in range(len(ids)):
            logger.debug(f" Generated text: {generated_text[i][len(self.structure):]}, id:{ids[i]}")
            if images[i] == "None" or images[i] is None:
                item = dict(id=ids[i],objects=generated_text[i][len(self.structure):],new_idx=new_idx[i])
            else:
                item = dict(id=ids[i],objects=generated_text[i][len(self.structure):],image=images[i],new_idx=new_idx[i])
            append_jsonl(item,self.output_file_path)
            
class LLaVATrainsetParser(ShareGPT4VTrainsetParser):
    def prepare_dataset(self):
        raw_data = load_json_file(self.input_file_path)
        messages = self.prepare_prompt()
        
        # self.data_dict = {i['id']:i for i in raw_data}
        self.meta_data = []
        logger.info(f"Start loading samples")
        for idx,item in enumerate(tqdm(raw_data)):
            new_idx = item["new_idx"]
            if self.processed_id.get(new_idx,False):
                continue
            
            image = item.get("image","None")
            messages_iter = deepcopy(messages)
            res_str = ""
            for conv_idx,conv in enumerate(item["conversations"]):
                res_str += f"{conv['from']}: {conv['value']}\n"
                
            messages_iter.append({"role": "user", "content": res_str})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":str(item['id']),"prompt":prompt,"image":image,"new_idx":new_idx})
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)