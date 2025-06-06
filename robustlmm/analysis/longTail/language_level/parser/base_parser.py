from mhr.utils.utils import process_jsonl,load_json_file, append_jsonl
import stanza
import os
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from tqdm import tqdm
import logging
import psycopg2
from time import sleep
from copy import deepcopy

# 创建一个logger

logger = logging.getLogger(__name__)



# nlp_pos = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
# text = 'This is token.ization done my way!\nSentence split, too!'
# doc = nlp_pos(text)
# print(doc)
# ALTER USER postgres WITH PASSWORD 'postgres';
def remove_duplicates(lst):
    return list(set(lst))

class BaseParser():
    def __init__(self,
                 dataset_file=None,
                 parser=stanza,
                 db_dict=None,
                 deduplicate=False,
                 output_file_path=None,
                 function=None) -> None:
        self.deduplicate = deduplicate
        assert db_dict is not None
        self.db_dict = db_dict
        
        if parser == stanza:
            self.parser=stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma',device='cpu')
            # stanza.Pipeline(lang='fr', processors='tokenize,pos',device='cpu')
        else:
            raise NotImplementedError
        
        self.function = function
        
        if function == "count_num":
            self.dataset = self.load_dataset(dataset_file)
            self.sentences = self.load_sentences()
            self.table_name = self.init_db()
        elif function == "object_extraction":
            self.output_file_path = output_file_path
            self.resume_from_file()
            self.dataset = self.load_dataset(dataset_file)
        else:
            raise NotImplementedError
            
        
        
    def resume_from_file(self):
        if os.path.exists(self.output_file_path):
            logger.info(f"checkpoint detected! Resume from file: {self.output_file_path}")
            self.cache = process_jsonl(self.output_file_path)
            self.processed_id = {item["id"]:1 for item in self.cache}
        else:
            os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
     
        
    def connect_db(self):
        conn = psycopg2.connect(**self.db_dict)
        cursor = conn.cursor()
        return conn,cursor
    

    def get_clean_str(self,input_str):
        return input_str.strip().replace("\n","").replace("\r","").replace("\t","").replace("'","_").replace('"',"_")
        
    def parse_sentence(self,sentence):
        llema_list = []
        doc = self.parser(sentence).to_dict()
        for sentence_parse in doc:
            for word in sentence_parse:
                xpos = word.get("xpos","none")
                if xpos.startswith('NN'):
                    lemma = word.get("lemma",None)
                    if lemma:
                        clean_lemma = self.get_clean_str(lemma)
                        llema_list.append(clean_lemma)
                    else:
                        logger.warning(f"lemma is None: {word}")
        if self.deduplicate:
            llema_list = remove_duplicates(llema_list)
        return llema_list
    
    def write_into_db(self,lemma_list):
        for lemma in lemma_list:
            self.write_item_into_db(lemma)
        
    def write_item_into_db(self,lemma, retry=10):
        if retry == 0:
            logger.error(f"Failed to write {lemma} into db")
            return
        try:
            conn,cursor = self.connect_db()
            logger.debug(f"SELECT count FROM {self.table_name} WHERE lemma='{lemma}'")
            cursor.execute(f"SELECT count FROM {self.table_name} WHERE lemma='{lemma}'")
            res = cursor.fetchall()
            assert len(res) < 2
            if len(res) == 1:
                count = res[0][0]
                count += 1
                logger.debug(f"UPDATE {self.table_name} SET count={count} WHERE lemma='{lemma}'")
                cursor.execute(f"UPDATE {self.table_name} SET count={count} WHERE lemma='{lemma}'")
            else:
                logger.debug(f"INSERT INTO {self.table_name} (lemma,count) VALUES ('{lemma}',1)")
                cursor.execute(f"INSERT INTO {self.table_name} (lemma,count) VALUES ('{lemma}',1)")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(e)
            sleep(1)
            self.write_item_into_db(lemma,retry-1)
            
    
    def extract_object(self,item):
        process_item = deepcopy(item)
        sentence = self.get_sentence_through_item(process_item)
        lemma_list = self.parse_sentence(sentence)
        lemma_list = remove_duplicates(lemma_list)
        process_item["objects"] = lemma_list
        append_jsonl(process_item,self.output_file_path)
        return 1 
    
    
    def parallel_extract(self,num_workers=32):
        logger.info(f"Start processing, num_workers={num_workers}")
        length = len(self.dataset)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(self.extract_object,self.dataset),total=length))
        
    
    
    def parallel_process(self,num_workers=32):
        def count_lemma(sentence):
            lemma_list = self.parse_sentence(sentence)   
            self.write_into_db(lemma_list)
            return 1
            
        logger.info(f"Start processing, num_workers={num_workers}")
        length = len(self.sentences)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(count_lemma,self.sentences),total=length))
        # for sentence in tqdm(self.sentences):
        #     count_lemma(sentence)
        
    def load_sentences(self):
        pass 
    
    def init_db(self):
        pass
    
    def get_sentence_through_item(self,item):
        pass
    
    def load_dataset(self, dataset_file):
        pass
                
        
    

class LCS558KParser(BaseParser): 
    def __init__(self,
                 dataset_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/blip_laion_cc_sbu_558k.json",
                 parser=stanza,
                 db_dict=dict(database='language_level_word_count',user='songmingyang',password='123456',host='SH-IDCA1404-10-140-54-108',port='5432'),
                 deduplicate=False,
                 output_file_path=None,
                 function=None) -> None:
        super().__init__(dataset_file,parser,db_dict,deduplicate,output_file_path,function)
    
    def load_sentences(self):
        sentences = []
        for item in self.dataset:
            conversations = item["conversations"]
            assert len(conversations) == 2
            sentence = conversations[1]['value']
            sentences.append(sentence)
        logger.info("Sentences loaded")
        return sentences

    def init_db(self):
        conn,cursor = self.connect_db()
        table_name = "LCS558K"
        if self.deduplicate:
            table_name += "_deduplicate"
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} (lemma TEXT PRIMARY KEY,count INTEGER)")
        conn.commit()
        conn.close()
        logger.info(f"Table {table_name} created")
        return table_name
    
    def get_sentence_through_item(self, item):
        conversations = item["conversations"]
        assert len(conversations) == 2
        sentence = conversations[1]['value']
        return sentence
    
    def load_dataset(self, dataset_file):
        dataset = load_json_file(dataset_file)
        res = []
        for item in dataset:
            if self.processed_id.get(item["id"],None) is not None:
                continue
            res.append(item)
        length = len(res)
        logger.info(f"Dataset loaded, length: {length}")
        return res
        
class Instructmix665KParser(BaseParser): 
    def __init__(self,
                 dataset_file="/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
                 parser=stanza,
                 db_dict=dict(database='language_level_word_count',user='songmingyang',password='123456',host='SH-IDCA1404-10-140-54-108',port='5432'),
                 deduplicate=False,
                 output_file_path=None,
                 function=None) -> None:
        super().__init__(dataset_file,parser,db_dict,deduplicate,output_file_path,function)
    
    def load_sentences(self):
        sentences = []
        for item in self.dataset:
            conversations = item["conversations"]
            res=""
            for conv in conversations:
                sentence = conv['value']
                res += sentence
            sentences.append(res)
        logger.info("Sentences loaded")
        return sentences

    def init_db(self):
        conn,cursor = self.connect_db()
        table_name = "Instructmix665K"
        if self.deduplicate:
            table_name += "_deduplicate"
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} (lemma TEXT PRIMARY KEY,count INTEGER)")
        conn.commit()
        conn.close()
        logger.info(f"Table {table_name} created")
        return table_name
    
    def get_sentence_through_item(self, item):
        conversations = item["conversations"]
        res=""
        for conv in conversations:
            sentence = conv['value']
            res += sentence
        return res
    
    
    def load_dataset(self, dataset_file):
        dataset = load_json_file(dataset_file)
        res = []
        for item in dataset:
            if self.processed_id.get(item["id"],None) is not None:
                continue
            res.append(item)
        length = len(res)
        logger.info(f"Dataset loaded, length: {length}")
        return res

class POPEParser(BaseParser): 
    def __init__(self,
                 dataset_file="/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/pope_mul/en_annotations",
                 parser=stanza,
                 db_dict=dict(database='language_level_word_count',user='songmingyang',password='123456',host='SH-IDCA1404-10-140-54-108',port='5432'),
                 deduplicate=False) -> None:
        super().__init__(dataset_file,parser,db_dict,deduplicate)
        
    
    def load_dataset(self, dataset_file):
        files = [os.path.join(dataset_file,f) for f in os.listdir(dataset_file)]
        res = []
        for file in files:
            file_data = process_jsonl(file)
            res+=file_data
        return res
    
    def load_sentences(self):
        sentences = []
        for item in self.dataset:
            sentence = item["text"]
            sentences.append(sentence)
        logger.info("Sentences loaded")
        return sentences

    def init_db(self):
        conn,cursor = self.connect_db()
        table_name = "POPE"
        if self.deduplicate:
            table_name += "_deduplicate"
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} (lemma TEXT PRIMARY KEY,count INTEGER)")
        conn.commit()
        conn.close()
        logger.info(f"Table {table_name} created")
        return table_name
    
class MMEParser(BaseParser): 
    def __init__(self,
                 dataset_file="/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/MME/MME_Benchmark_release_version",
                 parser=stanza,
                 db_dict=dict(database='language_level_word_count',user='songmingyang',password='123456',host='SH-IDCA1404-10-140-54-108',port='5432'),
                 deduplicate=False) -> None:
        super().__init__(dataset_file,parser,db_dict,deduplicate)
        
    
    def load_dataset(self, dataset_file):
        res = []
        categories = os.listdir(dataset_file)
        for category in categories:
            category_path = os.path.join(dataset_file,category)
            if not os.path.isdir(category_path) or "json_labels" not in os.listdir(category_path):
                continue
            base_file_name = os.path.join(category_path,"json_labels",f"mme_{category}_en.json")
            sentences = process_jsonl(base_file_name)
            res += sentences
        return res
    
    def load_sentences(self):
        sentences = []
        for item in self.dataset:
            sentence = item["text"]
            sentences.append(sentence)
        logger.info("Sentences loaded")
        return sentences

    def init_db(self):
        conn,cursor = self.connect_db()
        table_name = "MME"
        if self.deduplicate:
            table_name += "_deduplicate"
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} (lemma TEXT PRIMARY KEY,count INTEGER)")
        conn.commit()
        conn.close()
        logger.info(f"Table {table_name} created")
        return table_name
    
