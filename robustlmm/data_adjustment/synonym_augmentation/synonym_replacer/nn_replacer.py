from mhr.utils.utils import * 
import torch
import stanza
import logging
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

class NNReplacer(object):
    
    def __init__(self,args) -> None:
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.synonym_path = args.synonym_path
        self.distribution_reverse_index_files = args.distribution_reverse_index_files
        
        self.resume_from_ckpt()
        self.load_dataset()
        self.build_distribution_data()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parser=stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma',device=device)
        self.filter_words = [
                    # 抽象概念
                    "idea", "concept", "thought", "emotion", "feeling", "truth",
                    "time", "space", "fact", "question", "answer", "information",
                    "knowledge", "theory", "principle", "law", "rule", "method",
                    "system", "model", "process", "result", "problem", "issue",
                    "case", "situation", "condition", "event", "action", "activity",
                    "behavior", "experience", "practice", "skill", "ability",
                    "quality", "characteristic", "feature", "property", "attribute",
                    

                    # 时间、位置相关
                    "time", "moment", "second", "minute", "hour", "day", "week",
                    "month", "year", "location", "place", "area", "space",
                    "position", "direction", "distance", "point", "view", "side",
                    "front", "back", "top", "bottom", "middle", "end", "beginning",
                    "image", "picture", "scene",

                    # 颜色
                    "color", "shade", "hue",

                    # 虚拟或不可见的对象
                    "image", "video", "sound", "music", "voice", "software", 
                    "data", "text", "file",

                    # 情感和心理
                    "emotion", "thought", "feeling", "mood", "state", "condition",
                    "impression", "memory",

                    # 抽象的动词名词化形式
                    "decision", "suggestion", "explanation", "description", "action",
                    "reaction", "reflection", "process", "result", "change", "behavior",

                    # 人称代词和指示代词
                    "something", "anything", "nothing", "everything", "this", "that", 
                    "it", "one"
                ]
        self.filter_words = list(set(self.filter_words))
        
    def resume_from_ckpt(self):
        if os.path.exists(self.output_path):
            print_rank0(f"checkpoint detected! Resume from file: {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {str(item["new_idx"]):1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
             
    def load_dataset(self):
        raw_data = load_json_file(self.input_path)
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            new_idx = str(idx)
            if self.processed_id.get(new_idx,None) is not None:
                continue
            target_item = dict(new_idx=new_idx,conversations=item["conversations"])
            if item.get("image",None) is not None:
                target_item["image"] = item["image"]
            self.meta_data.append(target_item)
            
        synonym_data = process_jsonl(self.synonym_path)
        self.synonym_dict = {item["token"]:item["synonyms"] for item in synonym_data}
        
    def build_distribution_data(self):
        token_input_file,object_input_file,co_occurrence_input_file,what_word_input_file = self.distribution_reverse_index_files
        token_threshold,object_threshold,co_occurrence_threshold,what_word_threshold = [0.9]*4
        
        token_data = process_jsonl(token_input_file)
        object_data = process_jsonl(object_input_file)
        co_occurrence_data = process_jsonl(co_occurrence_input_file)
        what_word_data = process_jsonl(what_word_input_file)
        
        token_data.sort(key=lambda x: len(x["ids"]), reverse=True)
        object_data.sort(key=lambda x: len(x["ids"]), reverse=True)
        co_occurrence_data.sort(key=lambda x: len(x["ids"]), reverse=True)
        what_word_data.sort(key=lambda x: len(x["ids"]), reverse=True)
        
        token_data = [(x["object"],len(x["ids"])) for x in token_data]
        object_data = [(x["object"],len(x["ids"])) for x in object_data]
        co_occurrence_data = [(x["object"],len(x["ids"])) for x in co_occurrence_data]
        what_word_data = [(x["object"],len(x["ids"])) for x in what_word_data]
        
        

        token_sum = sum([x[1] for x in token_data])
        object_sum = sum([x[1] for x in object_data])
        co_occurrence_sum = sum([x[1] for x in co_occurrence_data])
        what_word_sum = sum([x[1] for x in what_word_data])
        
        def get_90_index(data,sum,ratio=0.9):
            sum_90 = sum*ratio
            sum_temp = 0
            for i in range(len(data)):
                sum_temp += data[i][1]
                if sum_temp >= sum_90:
                    return i
        

        token_90_loc = get_90_index(token_data,token_sum,token_threshold)
        object_90_loc = get_90_index(object_data,object_sum,object_threshold)
        co_occurrence_90_loc = get_90_index(co_occurrence_data,co_occurrence_sum,co_occurrence_threshold)
        what_word_90_loc = get_90_index(what_word_data,what_word_sum,what_word_threshold)
        print(f"token_{token_threshold}_loc:{token_90_loc}, object_{object_threshold}_loc:{object_90_loc}, co_occurrence_{co_occurrence_threshold}_loc:{co_occurrence_90_loc}, what_word_what_word_threshold_loc:{what_word_90_loc}")
        print(f"token_total {len(token_data)} object_total {len(object_data)} co_occurrence_total {len(co_occurrence_data)} what_word_total {len(what_word_data)}")
        
        
        self.token_dict = {x[0]:idx for idx,x in enumerate(token_data)}
        self.object_dict = {x[0]:idx for idx,x in enumerate(object_data)}
        self.co_occurrence_dict = {x[0]:idx for idx,x in enumerate(co_occurrence_data)}
        self.what_word_dict = {x[0]:idx for idx,x in enumerate(what_word_data)}
        self.loc_of_90=[token_90_loc,object_90_loc,co_occurrence_90_loc,what_word_90_loc]
    
    

    def get_clean_str(self,input_str):
        return input_str.strip().replace("\n","").replace("\r","").replace("\t","").replace("'","_").replace('"',"_")
        
    def extract_word_to_replace(self,sentence):
        llema_list = []
        doc = self.parser(sentence).to_dict()
        for sentence_parse in doc:
            for word in sentence_parse:
                xpos = word.get("xpos","none")
                if xpos.startswith('NN'):
                    lemma = word.get("lemma",None)
                    if lemma and lemma not in self.filter_words:
                        clean_lemma = self.get_clean_str(lemma)
                        llema_list.append(clean_lemma)

        return list(set(llema_list))
    
    def get_replace_word(self,to_replace):
        replace_info = []
        for replace_word in to_replace:
            candidates = self.synonym_dict.get(replace_word,[])
            pass_word = []
            for candidate in candidates:
                # assert isinstance(candidate,str),f"candidate is not str:{candidate}"
                candidate_str = candidate["synonym"]
                score = self.token_dict.get(candidate_str,-1)
                if score >= self.loc_of_90[0]:
                    pass_word.append(dict(origin=replace_word,replace=candidate_str,score=score))
            if len(pass_word) > 1:
                replace_info.append(sorted(pass_word,key=lambda x:x["score"],reverse=True)[0])
            elif len(pass_word) == 1:
                replace_info.append(pass_word[0])
        return replace_info
            
    
    def process_item(self,item):
        conversation = item["conversations"]
        new_conversation = []
        replace_num = 0
        for idx,conv in enumerate(conversation):
            from_token = conv["from"]
            sentence = conv["value"]
            to_replace = self.extract_word_to_replace(sentence)
            replace_info = self.get_replace_word(to_replace)
            new_sentence = sentence
            for info in replace_info:
                new_sentence = new_sentence.replace(info["origin"],info["replace"])
            new_conv = {"from":from_token,"value":new_sentence}
            new_conversation.append(new_conv)
            replace_num += len(replace_info)
        item["conversations"] = new_conversation
        item["old_conversations"] = conversation
        item["replace_num"] = replace_num
        self.write_output_item(item)
        return 1
    
    def sequential_process(self):
        for item in tqdm(self.meta_data):
            self.process_item(item)
    
    def parallel_process(self,max_workers=32):
        total = len(self.meta_data)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用线程池并发地执行任务，并创建一个进度条
            results = list(tqdm(executor.map(self.process_item, self.meta_data), total=total))
        return []
    
    def write_output_item(self,item):
        append_jsonl(item,self.output_path)
        

class ReplaceInfoExtractor(NNReplacer):
    def get_replace_word(self,to_replace):
        replace_info = []
        for replace_word in to_replace:
            candidates = self.synonym_dict.get(replace_word,[])
            pass_word = []
            for candidate in candidates:
                # assert isinstance(candidate,str),f"candidate is not str:{candidate}"
                candidate_str = candidate["synonym"]
                score = self.token_dict.get(candidate_str,-1)
                if score >= self.loc_of_90[0]:
                    pass_word.append(dict(origin=replace_word,replace=candidate_str,score=score))
            if len(pass_word) >= 1:
                replace_info.append({"token":replace_word,"candidates":pass_word})
        return replace_info
    
    def process_item(self,item):
        conversation = item["conversations"]
        sentence = ""
        for idx,conv in enumerate(conversation):
            sentence += conv["value"] + " "
        to_replace = self.extract_word_to_replace(sentence)
        replace_info = self.get_replace_word(to_replace)
        item["replace_info"] = replace_info
        item["to_replace"] = to_replace
        self.write_output_item(item)
        return 1