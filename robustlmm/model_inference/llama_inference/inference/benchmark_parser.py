from .base_inference import *

class VAL2014ExtractParser(LlamaParser):
    def resume_from_file(self):
        if os.path.exists(self.output_file_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {str(item["id"]):1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def prepare_dataset(self):
        raw_data = load_json_file(self.input_file_path)
        messages = self.prepare_prompt()
        
        extra_caption_data = process_jsonl(self.args.extra_caption_file)
        extra_caption_dict = {str(item["id"]):item["caption"] for item in extra_caption_data}
        main_caption_dict = {}
        for item in raw_data["annotations"]:
            if main_caption_dict.get(item['image_id']) is None:
                main_caption_dict[item['image_id']] = []
            main_caption_dict[item['image_id']].append(item['caption'])
        self.meta_data = []
        
        for item in raw_data["images"]:
            id = str(item["id"])
            if self.processed_id.get(id,False):
                continue
            image = item["file_name"]
            extra_caption = extra_caption_dict.get(id,"")
            main_caption = main_caption_dict.get(id,[])
            main_caption.append(extra_caption)
            caption = "".join(main_caption)
            if caption.strip() == "":
                continue
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": caption})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":id,"prompt":prompt,"image":image})
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
        
    def post_process_batch(self, batch):
        generated_text=batch["generated_text"]
        ids = batch["id"]
        images = batch["image"]
        for i in range(len(ids)):
            logger.debug(f" Generated text: {generated_text[i][len(self.structure):]}, id:{ids[i]}")
            item = dict(id=ids[i],objects=generated_text[i][len(self.structure):],image=images[i])
            append_jsonl(item,self.output_file_path)
            
            
class VQAV2WhatwordParser(VAL2014ExtractParser):
    
    def prepare_dataset(self):
        raw_data = load_json_file(self.input_file_path)["questions"]
        extra_answer_data = load_json_file(self.args.extra_answer_file)['annotations']
        extra_answer_dict = {str(item["question_id"]):item["answers"] for item in extra_answer_data}
        messages = self.prepare_prompt()
        self.meta_data = []
        for item in raw_data:
            id = str(item["question_id"])
            question = "human:"+item["question"]+"\n"
            image = str(item["image_id"])
            answers = extra_answer_dict.get(id,[])
            ans_str = "gpt:"
            answers = list(set([item["answer"] for item in answers]))
            for idx,ans in enumerate(answers):
                if idx == 0:
                    ans_str+=ans
                else:
                    ans_str+="/"+ans
            ans_str+="\n"
            res_str = question+ans_str
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": res_str})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":id,"prompt":prompt,"image":image})
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
        
    def post_process_batch(self, batch):
        generated_text=batch["generated_text"]
        ids = batch["id"]
        images = batch["image"]
        for i in range(len(ids)):
            logger.debug(f" Generated text: {generated_text[i][len(self.structure):]}, id:{ids[i]}")
            item = dict(id=ids[i],objects=generated_text[i][len(self.structure):],image_id=images[i])
            append_jsonl(item,self.output_file_path)

class SQAWhatwordParser(VAL2014ExtractParser):
    def resume_from_file(self):
        if os.path.exists(self.output_file_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {str(item["id"]):1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def prepare_dataset(self):
        raw_data = load_dataset(self.input_file_path,self.args.extra_subset_suffix)["test"]
        messages = self.prepare_prompt()
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            if self.processed_id.get(str(idx),False):
                continue
            question = item["question"]
            choices = " ".join(item["choices"])
            lecture = item["lecture"]
            solution = item["solution"]
            sentence = f"human:{question} {choices}\ngpt:{lecture} {solution}\n"
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": sentence})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":str(idx),"prompt":prompt})
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
        
    def post_process_batch(self, batch):
        generated_text=batch["generated_text"]
        ids = batch["id"]
        # images = batch["image"]
        for i in range(len(ids)):
            logger.debug(f" Generated text: {generated_text[i][len(self.structure):]}, id:{ids[i]}")
            item = dict(id=ids[i],objects=generated_text[i][len(self.structure):])
            append_jsonl(item,self.output_file_path)

class SQACaption2objectParser(SQAWhatwordParser):  
    def prepare_dataset(self):
        raw_data = process_jsonl(self.input_file_path)
        messages = self.prepare_prompt()
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            id = str(item["id"])
            if self.processed_id.get(id,False):
                continue
            
            caption = item["caption"]
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": caption})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":id,"prompt":prompt})
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
        
class POPELlamaParser(LlamaParser):
    def prepare_dataset(self):
        if os.path.isdir(self.input_file_path):
            raw_data = []
            for file in os.listdir(self.input_file_path):
                raw_data.extend(process_jsonl(os.path.join(self.input_file_path,file)))
        else:
            raw_data = process_jsonl(self.input_file_path)
        messages = self.prepare_prompt()
        self.data_dict = {i['id']:i for i in raw_data}
        self.meta_data = []
        logger.info(f"Start loading samples")
        for item in tqdm(raw_data):
            if self.processed_id.get(item['id'],False):
                continue
            
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": item["output"]})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)

            self.meta_data.append({"id":item['id'],"prompt":prompt})
            
        self.data_length = len(self.meta_data)
        logger.info(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
    def write_item_into_output_file(self,item):
        objects = item.pop("outputs")
        item["objects"] = objects
        append_jsonl(item,self.output_file_path)