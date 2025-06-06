from mhr.utils.utils import *


    
class BaseEvaluator(object):
    def __init__(self,args):
        self.statistic_files = args.statistic_files
        self.statistic_id_key = args.statistic_id_key
        self.statistic_object_keys = args.statistic_object_keys
        
        self.distribution_compose_list = args.distribution_compose_list
        self.distribution_reverse_index_files = args.distribution_reverse_index_files
        self.distribution_thresholds = args.distribution_thresholds

        self.head_tail_pass_key = args.head_tail_pass_key
        
        self.result_file_name = args.result_file_name
        self.result_question_id_key = args.result_question_id_key
        self.result_score_key = args.result_score_key
        self.result_files = args.result_files
        self.result_names = args.result_names
        
    def build_statistic_data(self):
        assert isinstance(self.statistic_files,list)
        if len(self.statistic_files) == 1:
            if self.statistic_files[0].endswith(".jsonl"):
                self.statistic_data = process_jsonl(self.statistic_files[0])
            elif self.statistic_files[0].endswith(".json"):
                self.statistic_data = load_json_file(self.statistic_files[0])
        else:
            token_data = process_jsonl(self.statistic_files[0])
            object_data = process_jsonl(self.statistic_files[1])
            what_word_data = process_jsonl(self.statistic_files[2])
            
            object_dict = {item[self.statistic_id_key]:str2list(item[self.statistic_object_keys[1]]) for item in object_data}
            token_dict = {item[self.statistic_id_key]:str2list(item[self.statistic_object_keys[0]]) for item in token_data}
            what_word_dict = {item[self.statistic_id_key]:str2list(item[self.statistic_object_keys[2]]) for item in what_word_data}
            
            self.statistic_data = []
            ids = list(set(list(object_dict.keys()) + list(token_dict.keys()) + list(what_word_dict.keys())))
            for id in ids:
                objects = list(set(object_dict.get(id,[])))
                tokens = token_dict.get(id,[])
                what_words = what_word_dict.get(id,[])
                co_occurrences = []
                for i in range(len(objects)):
                    for j in range(i+1,len(objects)):
                        two_words = get_two_words(objects[i],objects[j])
                        co_occurrences.append(two_words)
                target_dict=dict(id=id,statistics=dict(object=objects,token=tokens,what_word=what_words,co_occurrence=co_occurrences))
                self.statistic_data.append(target_dict)
    
    def build_distribution_data(self):
        token_input_file,object_input_file,co_occurrence_input_file,what_word_input_file = self.distribution_reverse_index_files
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
        token_threshold,object_threshold,co_occurrence_threshold,what_word_threshold = self.distribution_thresholds

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
    
    
    def get_head_tail_data_ids(self):
        def lookup_to_dict(obj,dictionary):
            return dictionary.get(obj,-1)

        def get_scores(data,dictionary,threshold):
            scores = []
            metadata = []
            for item in data:
                score = lookup_to_dict(item,dictionary)
                if score >= 0:
                    scores.append(score)
                    metadata.append({"object":item,"score":score})
            avg_score = sum(scores)/len(scores) if len(scores) > 0 else 0
            max_score = max(scores) if len(scores) > 0 else 0
            avg_pass = avg_score >= threshold
            least_pass = max_score >= threshold
            return scores,metadata,avg_pass,least_pass
        statistic_data = self.statistic_data
        token_90_loc,object_90_loc,co_occurrence_90_loc,what_word_90_loc = self.loc_of_90
        
        for item in statistic_data:
            tokens = item["statistics"]["token"]
            objects = item["statistics"]["object"]
            co_occurrences = item["statistics"]["co_occurrence"]
            what_words = item["statistics"]["what_word"]
            
            token_scores,token_metadata,token_avg_pass,token_least_pass = get_scores(tokens,self.token_dict,token_90_loc)
            object_scores,object_metadata,object_avg_pass,object_least_pass = get_scores(objects,self.object_dict,object_90_loc)
            co_occurrence_scores,co_occurrence_metadata,co_occurrence_avg_pass,co_occurrence_least_pass = get_scores(co_occurrences,self.co_occurrence_dict,co_occurrence_90_loc)
            what_word_scores,what_word_metadata,what_word_avg_pass,what_word_least_pass = get_scores(what_words,self.what_word_dict,what_word_90_loc)
            distribution_item = dict(token=dict(scores=token_scores,metadata=token_metadata,avg_pass=token_avg_pass,least_pass=token_least_pass),
                                    object=dict(scores=object_scores,metadata=object_metadata,avg_pass=object_avg_pass,least_pass=object_least_pass),
                                    co_occurrence=dict(scores=co_occurrence_scores,metadata=co_occurrence_metadata,avg_pass=co_occurrence_avg_pass,least_pass=co_occurrence_least_pass),
                                    what_word=dict(scores=what_word_scores,metadata=what_word_metadata,avg_pass=what_word_avg_pass,least_pass=what_word_least_pass))
            item["distribution"] = distribution_item
        
        token_pass_cnt = 0
        object_pass_cnt = 0
        co_occurrence_pass_cnt = 0
        what_word_pass_cnt = 0
        all_pass_cnt = 0
        pass_key = self.head_tail_pass_key
        pass_ids = []
        not_pass_ids = []
        for item in statistic_data:
            token_pass,object_pass,co_occurrence_pass,what_word_pass = 0,0,0,0
            if item["distribution"]["token"][pass_key]:
                token_pass = 1
                token_pass_cnt += 1
            if item["distribution"]["object"][pass_key]:
                object_pass = 1
                object_pass_cnt += 1
            if item["distribution"]["co_occurrence"][pass_key]:
                co_occurrence_pass_cnt += 1
                co_occurrence_pass = 1
            if item["distribution"]["what_word"][pass_key]:
                what_word_pass_cnt += 1
                what_word_pass = 1
            if token_pass + object_pass + co_occurrence_pass + what_word_pass >= 1:
                all_pass_cnt += 1
                pass_ids.append(item["id"])
            else:
                not_pass_ids.append(item["id"])
        self.pass_ids = pass_ids
        self.not_pass_ids = not_pass_ids
        self.tail_dict = {i:1 for i in pass_ids}
        self.head_dict = {i:1 for i in not_pass_ids}
        print(f"token_pass_cnt:{token_pass_cnt}, object_pass_cnt:{object_pass_cnt}, \
      co_occurrence_pass_cnt:{co_occurrence_pass_cnt}, what_word_pass_cnt:{what_word_pass_cnt}, all_pass_cnt:{all_pass_cnt}")
        
        
    
    def build_eval_result_data(self):
        def get_item_by_multi_index(dictionary,keys):
            target = dictionary
            for key in keys:
                target = target[key]
            return target
        
        def get_res_from_lmms_eval(result_file):
            if os.path.isdir(result_file):
                result_file = os.path.join(result_file,self.result_file_name)
            else:
                result_file = result_file
            return load_json_file(result_file)["logs"]

        def get_result(result_data,head_dict,tail_dict):
            head_result = []
            tail_result = []
            for item in result_data:
                question_id = str(get_item_by_multi_index(item,self.result_question_id_key))
                if head_dict.get(question_id,-1) >= 0:
                    head_result.append(item)
                elif tail_dict.get(question_id,-1) >= 0:
                    tail_result.append(item)
            assert len(head_result) > 0
            assert len(tail_result) > 0
            head_scores = [get_item_by_multi_index(x,self.result_score_key) for x in head_result]
            tail_scores = [get_item_by_multi_index(x,self.result_score_key) for x in tail_result]
            head_acc = sum(head_scores)/len(head_scores)
            tail_acc = sum(tail_scores)/len(tail_scores)
            return head_acc,tail_acc

        result_data = []
        for i,result_file in enumerate(self.result_files):
            temp_result = get_res_from_lmms_eval(result_file)
            head_acc,tail_acc = get_result(temp_result,self.head_dict,self.tail_dict)
            result_data.append(dict(name=self.result_names[i],head_acc=head_acc,tail_acc=tail_acc))
        self.result_data = result_data
    
    def print_res(self):
        for item in self.result_data:
            print(f"Model:{item['name']} head_acc:{item['head_acc']} tail_acc:{item['tail_acc']}")
    
    def full_procedure(self):
        self.build_statistic_data()
        self.build_distribution_data()
        self.get_head_tail_data_ids()
        self.build_eval_result_data()
        self.print_res()