from mhr.utils.utils import process_jsonl,write_jsonl
from tqdm import tqdm

from transformers import  HfArgumentParser
from tqdm import tqdm
from dataclasses import dataclass, field



def str2list(input_str):
    if isinstance(input_str,str):
        raw_list = input_str.strip().replace("\n","").split(",")
        new_list = []
        for item in raw_list:
            new_list.append(item.strip())
        return new_list
    elif isinstance(input_str,list):
        return input_str
    else:
        raise TypeError("input_str should be str or list")

def get_two_words(word1,word2):
    if word1 < word2:
        return f"{word1},{word2}"
    else:
        return f"{word2},{word1}"
    

class ReverseIndexer(object):
    def __init__(self,
                 args
                 ) -> None:
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.id_key = args.id_key
        self.load_data()
    
    def load_data(self) -> None:
        self.meta_data = process_jsonl(self.input_path)
    
    def write_data(self,data):
        write_jsonl(data,self.output_path)
    
    def reverse_indexing(self):
        pass


    

class ObjectLLAMAReverseIndexer(ReverseIndexer):
    def reverse_indexing(self):
        reverse_indexed_dict={}
        for item in tqdm(self.meta_data):
            objects= list(set(str2list(item["objects"])))
            id = item[self.id_key]
            for object in objects:
                if object not in reverse_indexed_dict:
                    reverse_indexed_dict[object]=[]
                reverse_indexed_dict[object].append(id)
        
        reverse_indexed_list = [dict(object=object,ids=ids,num_ids=len(ids)) for object,ids in reverse_indexed_dict.items()]
        reverse_indexed_list.sort(key=lambda x: len(x["ids"]), reverse=True)
        self.write_data(reverse_indexed_list)
        

class Co_occurReverseIndexer(ReverseIndexer):
    
    def reverse_indexing(self):
        reverse_indexed_dict={}
        for item in tqdm(self.meta_data):
            objects= list(set(str2list(item["objects"])))
            for i in range(len(objects)):
                for j in range(i+1,len(objects)):
                    key = get_two_words(objects[i],objects[j])
                    if reverse_indexed_dict.get(key) is None:
                        reverse_indexed_dict[key]=[]
                    reverse_indexed_dict[key].append(item[self.id_key])
        reverse_indexed_list = [dict(object=object,ids=ids,num_ids=len(ids)) for object,ids in reverse_indexed_dict.items()]
        reverse_indexed_list.sort(key=lambda x: len(x["ids"]), reverse=True)
        self.write_data(reverse_indexed_list)
        
        
class TokenReverseIndexer(ReverseIndexer):
    def reverse_indexing(self):
        reverse_indexed_dict={}
        for item in tqdm(self.meta_data):
            objects= str2list(item["tokens"])
            id = item[self.id_key]
            for object in objects:
                if object not in reverse_indexed_dict:
                    reverse_indexed_dict[object]=[]
                reverse_indexed_dict[object].append(id)
        
        reverse_indexed_list = [dict(object=object,ids=ids,num_ids=len(ids)) for object,ids in reverse_indexed_dict.items()]
        reverse_indexed_list.sort(key=lambda x: len(x["ids"]), reverse=True)
        self.write_data(reverse_indexed_list)

class WhatWordReverseIndexer(ReverseIndexer):
    def reverse_indexing(self):
        reverse_indexed_dict={}
        test_item = self.meta_data[0]
        if test_item.get("what_words") is not None:
            what_word_key = "what_words"
        elif test_item.get("objects") is not None:
            what_word_key = "objects"
        elif test_item.get("outputs") is not None:
            what_word_key = "outputs"
        else:
            raise ValueError("what_words or objects should be in the meta_data")
        for item in tqdm(self.meta_data):
            objects= list(set(str2list(item.get("objects",item.get("outputs","")))))
            id = item[self.id_key]
            for object in objects:
                if object not in reverse_indexed_dict:
                    reverse_indexed_dict[object]=[]
                reverse_indexed_dict[object].append(id)
        
        reverse_indexed_list = [dict(object=object,ids=ids,num_ids=len(ids)) for object,ids in reverse_indexed_dict.items()]
        reverse_indexed_list.sort(key=lambda x: len(x["ids"]), reverse=True)
        self.write_data(reverse_indexed_list)


function_dict={
    "llamaobj":ObjectLLAMAReverseIndexer,
    "llamacooccur":Co_occurReverseIndexer,
    "token":TokenReverseIndexer,
    "whatword":WhatWordReverseIndexer,
    "dinoobj":ObjectLLAMAReverseIndexer,
    "dinocooccur":Co_occurReverseIndexer,
}

if __name__ == "__main__":
        
    @dataclass
    class InferenceArguments:
        input_path: str = field(default="/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back")
        output_path: str = field(default="/mnt/petrelfs/songmingyang/quxiaoye/models/Meta-Llama-3-70B-Instruct-back")
        function: str = field(default="llamaobj")
        id_key : str = field(default="id")
        
        
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]
    function_instance = function_dict[args.function](args)
    function_instance.reverse_indexing()
    
    