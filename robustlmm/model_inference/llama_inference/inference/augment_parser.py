from .base_inference import *

class BaseAugmentParser(LlamaParser):
    def check_and_filter(self,item):
        origin_output = item.pop("outputs")
        raw_conv_list=origin_output.strip().split("===")
        if "Question:" not in raw_conv_list[0]:
            # print(f"item_id:{item_idx} item: {raw_conv_list[0]}")
            return
        if not raw_conv_list[0].startswith("Question:"):
            loc = raw_conv_list[0].find("Question:")
            raw_conv_list[0] = raw_conv_list[0][loc-1:]
        for raw_conv_idx,raw_conv in enumerate(raw_conv_list):
            if "Question:" in raw_conv and "Answer:" in raw_conv:
                q_start = raw_conv.find("Question:")
                a_start = raw_conv.find("Answer:")
                insert_loc = max(q_start,a_start)
                front_str = raw_conv[:insert_loc]
                back_str = raw_conv[insert_loc:]
                raw_conv_list[raw_conv_idx]=back_str
                raw_conv_list.insert(raw_conv_idx,front_str)
        conv_list=[]
        for raw_conv_idx,raw_conv in enumerate(raw_conv_list):
            if raw_conv_idx %2 == 0:
                if raw_conv.startswith("\nQuestion:"):
                    conv_list.append({"from":"human","value":raw_conv[len('\nQuestion:'):]})
                elif raw_conv.startswith("Question:"):
                    conv_list.append({"from":"human","value":raw_conv[len('Question:'):]})
                else:
                    return
            else:
                if raw_conv.startswith("\nAnswer:"):
                    conv_list.append({"from":"gpt","value":raw_conv[len('\nAnswer:'):]})
                elif raw_conv.startswith("Answer:"):
                    conv_list.append({"from":"gpt","value":raw_conv[len('Answer:'):]})
                else:
                    return
        image_token_num = 0
        for conv in conv_list:
            image_token_num += conv["value"].count("<image>")
        if item.get("image",None) is None:
            if image_token_num > 0:
                for conv in conv_list:
                    conv["value"]=conv["value"].replace("<image>","")
        else:
            if image_token_num > 1:
                for conv in conv_list:
                    conv["value"]=conv["value"].replace("<image","")
                conv_list[0]["value"] = "<image>\n"+conv_list[0]["value"] 
            elif image_token_num == 0:
                conv_list[0]["value"] = "<image>\n"+conv_list[0]["value"] 
        
        item["conversations"]=conv_list
        new_output_str=""
        for raw_conv_idx,raw_conv in enumerate(raw_conv_list):
            if raw_conv_idx == len(raw_conv_list)-1:
                new_output_str+=raw_conv
            else:
                new_output_str+=raw_conv+"==="
        item["origin_output"]=new_output_str
        return item

class TokenAugmentParser(BaseAugmentParser):
    def resume_from_file(self):
        if os.path.exists(self.output_file_path):
            logger.info(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {item["new_idx"]:1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
            
    def prepare_dataset(self):
        raw_data = load_json_file(self.input_file_path)     
        synonym_data = process_jsonl(self.args.synonym_file)
        synonym_dict = {item["token"]:item["synonyms"] for item in synonym_data}
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            if self.processed_id.get(str(idx),False):
                continue
            res_str = "Conversation:\n[\n"
            for conv_idx,conv in enumerate(item["conversations"]):
                if conv["from"] == "human":
                    res_str += f"Question: {conv['value']}\n"
                elif conv["from"] == "gpt":
                    res_str += f"Answer: {conv['value']}\n"
                else:
                    raise ValueError(f"Unrecognized conversation: {conv}")
                
            res_str += "\n]\n===\nCandidate words:\n["
            candidates = []
            tokens = item["statistics"]["token"]
            for token in tokens:
                synonyms = synonym_dict.get(token,[])
                for synonym in synonyms:
                    if synonym["value"] > 0:
                        candidates.append(synonym["synonym"])
            # if len(candidates) < 1:
            #     continue
            for word_idx,word in enumerate(candidates):
                if word_idx == 0:
                    res_str += f"{word}"
                else:
                    res_str += f", {word}"
            res_str += "]\n"
            # print(f"res_str: {res_str}")
            messages = deepcopy(self.prepare_prompt())
            messages.append({"role": "user", "content": res_str})
            prompt = self.tokenizer.apply_chat_template(messages,tokenize=False)
            meta_item={"id":str(item['id']),"prompt":prompt,"new_idx":str(idx),"image":item.get("image","None")}
            self.meta_data.append(meta_item)
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
    
    def post_process_batch(self, batch):
        generated_text=batch["generated_text"]
        ids = batch["id"]
        new_idx = batch["new_idx"]
        images = batch.get("image",[None]*len(ids))
        for i in range(len(ids)):
            logger.debug(f" Generated text: {generated_text[i][len(self.structure):]}, id:{ids[i]}")
            if images[i] == "None" or images[i] is None:
                item = dict(id=ids[i],outputs=generated_text[i][len(self.structure):],new_idx=new_idx[i])
            else:
                item = dict(id=ids[i],outputs=generated_text[i][len(self.structure):],image=images[i],new_idx=new_idx[i])
            checked_item=self.check_and_filter(item)
            if checked_item is not None:
                append_jsonl(checked_item,self.output_file_path)

class TailTokenLMReParaphrase(TokenAugmentParser):
    def prepare_dataset(self):
        raw_data = load_json_file(self.input_file_path)     
        # synonym_data = process_jsonl(self.args.synonym_file)
        # synonym_dict = {item["token"]:item["synonyms"] for item in synonym_data}
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            if self.processed_id.get(str(idx),False):
                continue
            res_str = ""
            for conv_idx,conv in enumerate(item["conversations"]):
                if conv["from"] == "human":
                    res_str += f"Question: {conv['value']}\n"
                elif conv["from"] == "gpt":
                    res_str += f"Answer: {conv['value']}\n"
                else:
                    raise ValueError(f"Unrecognized conversation: {conv}")
            # print(f"res_str: {res_str}")
            messages = deepcopy(self.prepare_prompt())
            messages.append({"role": "user", "content": res_str})
            prompt = self.tokenizer.apply_chat_template(messages,tokenize=False)
            meta_item={"id":str(item['id']),"prompt":prompt,"new_idx":str(idx),"image":item.get("image","None")}
            self.meta_data.append(meta_item)
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)

class PlainAugCaption2instanceParser(BaseAugmentParser):
    def prepare_dataset(self):
        raw_data = process_jsonl(self.input_file_path)
        messages = self.prepare_prompt()
        self.meta_data = []
        for item in raw_data:
            if self.processed_id.get(item['id'],False):
                continue
            
            messages_iter = deepcopy(messages)
            messages_iter.append({"role": "user", "content": item["caption"]})
            prompt = self.tokenizer.apply_chat_template(messages_iter,tokenize=False)
            self.meta_data.append({"id":item['id'],"prompt":prompt,"image":item["image"]})
            
        self.data_length = len(self.meta_data)
        print_rank0(f"Finish loading samples, total samples: {self.data_length}")
        self.meta_data = ray.data.from_items(self.meta_data)
    
    def post_process_batch(self, batch):
        generated_text=batch["generated_text"]
        ids = batch["id"]
        images = batch["image"]
        for i in range(len(ids)):
            logger.debug(f" Generated text: {generated_text[i][len(self.structure):]}, id:{ids[i]}")
            item = dict(id=ids[i],outputs=generated_text[i][len(self.structure):],image=images[i])
            checked_item=self.check_and_filter(item)
            if checked_item is not None:
                append_jsonl(checked_item,self.output_file_path)
        
class AugCaptionLlamaParser(PlainAugCaption2instanceParser):
        pass




            
    
        
