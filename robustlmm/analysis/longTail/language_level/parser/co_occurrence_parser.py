from neo4j import GraphDatabase
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from mhr.utils.utils import process_jsonl, load_json_file, append_jsonl
import os
import logging

insert_node_query="""
MERGE (n:WORD {{ lemma: '{lemma}' }})
ON CREATE SET n.count = 1
ON MATCH SET n.count = n.count + 1;
"""

insert_relation_query="""
MERGE (a:WORD {{ lemma:'{lemma1}' }})
ON CREATE SET a.count = 1
MERGE (b:WORD {{ lemma:'{lemma2}' }})
ON CREATE SET b.count = 1
MERGE (a)-[r:CO_OCCURRENCE]->(b)
ON CREATE SET r.count = 0
WITH a,b,r
SET r.count = r.count + 1;
"""
logger = logging.getLogger(__name__)

class CoOccurrenceBaseParser():
    
    def __init__(
        self,
        neo4j_dict = None,
        object_file_path = None,
        ckpt_file_path = None,
    ):
        self.object_file_path = object_file_path
        self.neo4j_dict = neo4j_dict
        self.database_name = self.neo4j_dict.get("database_name",None)
        self.ckpt_file_path = ckpt_file_path
        
        self.connect_to_neo4j()
        self.resume_from_ckpt()
        self.load_data()
    
    
    def resume_from_ckpt(self):
        assert self.ckpt_file_path is not None
        if os.path.exists(self.ckpt_file_path):
            logger.info(f"checkpoint detected! Resume from file: {self.ckpt_file_path}")
            self.cache = process_jsonl(self.ckpt_file_path)
            self.processed_id = {item["id"]:True for item in self.cache }
        else:
            os.makedirs(os.path.dirname(self.ckpt_file_path),exist_ok=True)
            self.cache = []
            self.processed_id = {}
    
    def load_data(self):
        assert self.object_file_path is not None
        raw_data = process_jsonl(self.object_file_path)
        self.data = []
        for item in raw_data:
            if self.processed_id.get(item["id"],False):
                continue
            self.data.append(item)
        logger.info(f"Finish loading samples, total samples: {len(self.data)}")
            
    
    def connect_to_neo4j(self):
        
        driver = GraphDatabase.driver(self.neo4j_dict.get("uri","neo4j://10.140.54.61:7687"),
                                      auth=(self.neo4j_dict.get("username","neo4j"), self.neo4j_dict.get("password","12345678")))
        self.driver = driver
    
    def process_object_item(self,object_item):
        if isinstance(object_item,list):
            return object_item
        elif isinstance(object_item, str):
            return object_item.strip().replace("'","").replace("\\","").replace('"',"").split(",")
    
    def get_2_lemma_in_order(self,lemma1,lemma2):
        assert lemma1 != lemma2
        if lemma1 < lemma2:
            return lemma1,lemma2
        else:
            return lemma2,lemma1
    
    def write_one_sentence_into_neo4j(self,item):
        object_item = item["objects"]
        object_item = self.process_object_item(object_item)
        object_item = list(set(object_item))
        length = len(object_item)
        for object_word in object_item:
            try:
                with self.driver.session(database=self.database_name) as session:
                    session.run(insert_node_query.format(lemma=object_word))
            except Exception as e:
                logger.error(f"Error in inserting node {object_word}, error: {e}")
        for i in range(length):
            for j in range(i+1,length):
                try:
                    with self.driver.session(database=self.database_name) as session:
                        lemma1,lemma2 = self.get_2_lemma_in_order(object_item[i],object_item[j])
                        session.run(insert_relation_query.format(lemma1=lemma1,lemma2=lemma2))
                except Exception as e:
                    logger.error(f"Error in inserting relation {object_item[i]} and {object_item[j]}, error: {e}")
        append_jsonl({"id":item["id"]},self.ckpt_file_path)
        return 1
    
    def parallel_process(self,num_workers=32):
        length = len(self.data)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(self.write_one_sentence_into_neo4j,self.data),total=length))
    

class CoOccurrenceVisualParser(CoOccurrenceBaseParser):
    def load_data(self):
        assert self.object_file_path is not None
        raw_data = process_jsonl(self.object_file_path)
        self.data = []
        for item in raw_data:
            if self.processed_id.get(item["id"],False):
                continue
            item["objects"] = item["statistic"]["labels"]
            self.data.append(item)
        logger.info(f"Finish loading samples, total samples: {len(self.data)}")