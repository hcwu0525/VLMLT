from mhr.utils.utils import process_jsonl, load_json_file, write_json_file
import random
import math
from tqdm import tqdm
import argparse


def sample_data(threshold_dict, reverse_index_file_dict, input_dataset_file_path ,output_file_path,pass_num=1):
    """
    Sample n rows from data.
    """
    D_star=[]
    entry_prob = build_prob_dict(reverse_index_file_dict,threshold_dict)
    origin_data = load_json_file(input_dataset_file_path)
    
    
    for item in tqdm(origin_data):
        pass_cnt = 0
        for key in ['object','token','co_occurrence','what_word']:
            for obj in item['statistics'][key]:
                prob = entry_prob[key].get(obj,0)
                if random.random() < prob:
                    pass_cnt += 1
                    break
        if pass_cnt > pass_num:
            item.pop('statistics')
            D_star.append(item)
    write_json_file(D_star,output_file_path)


def sample_data_compose(threshold_dict, reverse_index_file_dict, input_dataset_file_path ,output_file_path,compose_list,pass_num=0):
    """
    Sample n rows from data.
    """
    D_star=[]
    entry_prob = build_prob_dict(reverse_index_file_dict,threshold_dict)
    origin_data = load_json_file(input_dataset_file_path)
    assert isinstance(compose_list,list)
    
    for item in tqdm(origin_data):
        pass_cnt = 0
        for key in compose_list:
            for obj in item['statistics'][key]:
                prob = entry_prob[key].get(obj,0)
                if random.random() < prob:
                    pass_cnt += 1
                    break
        if pass_cnt > pass_num:
            item.pop('statistics')
            D_star.append(item)
    print(f"length of D_star: {len(D_star)}")
    write_json_file(D_star,output_file_path)
    
    
def sample_data_add(threshold_dict, reverse_index_file_dict, input_dataset_file_path ,output_file_path,pass_num=1):
    """
    Sample n rows from data.
    """
    D_star=[]
    entry_prob = build_prob_dict(reverse_index_file_dict,threshold_dict)
    origin_data = load_json_file(input_dataset_file_path)
    
    pass_cnt_list = []
    for item in tqdm(origin_data):
        max_of_prob = 0
        pass_dict = dict(object=False, token=False, co_occurrence=False, what_word=False)
        for key in ['object','token','co_occurrence','what_word']:
            for obj in item['statistics'][key]:
                prob = entry_prob[key].get(obj,0)
                if prob > max_of_prob:
                    max_of_prob = prob
                if random.random() < prob:
                    pass_dict[key] = True
        pass_cnt = sum(pass_dict.values())
        sqrt_of_prob = int(math.sqrt(max_of_prob))
        enroll_cnt = sqrt_of_prob if sqrt_of_prob > 1 else 1
        enroll_cnt = enroll_cnt if enroll_cnt < 3 else 3
        # enroll_cnt = int(max_of_prob) if int(max_of_prob) < 5  else 5
        pass_cnt_list.append(enroll_cnt)
        if pass_cnt > pass_num:
            item.pop('statistics')
            D_star.extend([item]*(enroll_cnt))
    write_json_file(D_star,output_file_path)
    print(f"length of D_star: {len(D_star)}")
    print(f"average enroll_cnt: {sum(pass_cnt_list)/len(pass_cnt_list)}")

def sample_data_one(threshold_dict, reverse_index_file_dict, input_dataset_file_path ,output_file_path,pass_num=1,key="token"):
    D_star=[]
    entry_prob = build_prob_dict(reverse_index_file_dict,threshold_dict)
    origin_data = load_json_file(input_dataset_file_path)
    
    pass_cnt_list = []
    for item in tqdm(origin_data):
        max_of_prob = 0
        pass_flag = False
        # for key in ['object','token','co_occurrence','what_word']:
        for obj in item['statistics'][key]:
            prob = entry_prob[key].get(obj,0)
            # assert prob is not None, f"key: {key}, obj: {obj}"
            
            if random.random() < prob:
                pass_flag = True
                break

        if pass_flag:
            item.pop('statistics')
            D_star.append(item)
    write_json_file(D_star,output_file_path)
    print(f"length of D_star: {len(D_star)}")
    # print(f"average enroll_cnt: {sum(pass_cnt_list)/len(pass_cnt_list)}")
    

def build_prob_dict(file_dict,threshold_dict):
    """
    Build a dictionary of probabilities for each entry in the data.
    """
    entry_prob={}
    for key in ['object','token','co_occurrence','what_word']:
        entry_prob[key] = dict()
        data = process_jsonl(file_dict[key]) 
        for item in data:
            # length = len(item['ids']) if len(item['ids']) > threshold_dict[key] else threshold_dict[key]
            length = len(item['ids']) 
            entry_prob[key][item['object']] = threshold_dict[key] / length 

    return entry_prob

def sample_data_compose_alpha(threshold_dict, reverse_index_file_dict, input_dataset_file_path ,output_file_path,compose_list,alpha,pass_num=0):
    """
    Sample n rows from data.
    """
    D_star=[]
    entry_prob = build_prob_dict(reverse_index_file_dict,threshold_dict)
    origin_data = load_json_file(input_dataset_file_path)
    assert isinstance(compose_list,list)
    
    for item in tqdm(origin_data):
        pass_cnt = 0
        for key in compose_list:
            for obj in item['statistics'][key]:
                prob = entry_prob[key].get(obj,0)
                if random.random() < prob:
                    pass_cnt += 1
                    break
        if pass_cnt > pass_num and random.random() < alpha:
            item.pop('statistics')
            D_star.append(item)
    print(f"length of D_star: {len(D_star)}")
    write_json_file(D_star,output_file_path)


def sample_data_compose_alpha_average(threshold_dict, reverse_index_file_dict, input_dataset_file_path ,output_file_path,compose_list,alpha,pass_num=0):
    """
    Sample n rows from data.
    """
    D_star=[]
    entry_prob = build_prob_dict(reverse_index_file_dict,threshold_dict)
    origin_data = load_json_file(input_dataset_file_path)
    assert isinstance(compose_list,list)
    
    for item in tqdm(origin_data):
        pass_cnt = 0
        for key in compose_list:
            # for obj in item['statistics'][key]:
            #     prob = entry_prob[key].get(obj,0)
            prob_list = [entry_prob[key].get(obj,0) for obj in item['statistics'][key]]
            avg_prob = sum(prob_list)/len(prob_list)
            if random.random() < avg_prob:
                pass_cnt += 1
        if pass_cnt > pass_num and random.random() < alpha:
            item.pop('statistics')
            D_star.append(item)
    print(f"length of D_star: {len(D_star)}")
    write_json_file(D_star,output_file_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_file",type=str,help="output file",default=None)
    argparser.add_argument("--input_dataset_file",type=str,help="input file",default=None)
    argparser.add_argument("--pass_num",type=int,help="input file",default=0)
    # argparser.add_argument("--add",action='store_true',help="input file",default=False)
    argparser.add_argument("--seed",type=int,help="threshold dict",default=42)
    argparser.add_argument("--mode",default="sample",type=str,help="mode")
    argparser.add_argument("--key",default=None,type=str,help="key")
    argparser.add_argument("--compose_list",default=None,type=str,help="compose_list",nargs='*')
    argparser.add_argument("--alpha",default=None,type=float,help="alpha")
    argparser.add_argument("--target_model",default="llava",type=str,help="target_model")
    args=argparser.parse_args()
    if args.target_model == "llava_ft":
        threshold_dict = {'object': 304, 'token': 120, 'co_occurrence': 24, 'what_word': 4895}
        reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_dino_stat_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_co_occurrence_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_what_word_reverse_index.jsonl'
        }
    elif args.target_model == "llava_ft_new":
        threshold_dict = {'object': 211, 'token': 74, 'co_occurrence': 11, 'what_word': 194}
        reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/dinoobj_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/dinocooccur_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/new/reverse_index/whatword_reverse_index.jsonl'
        }
    elif args.target_model == "sharegpt4v_pt":
        threshold_dict = {'token': 2459, 'object': 107, 'co_occurrence': 12, 'what_word': 4895}
        threshold_dict = {'token': 633, 'object': 57, 'co_occurrence': 12, 'what_word': 4895}
        reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/reverse_index/dinoobj_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/reverse_index/token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/pt/reverse_index/dinocooccur_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-Instruct-150K/reverse_index/llava_v1_5_mix665k_what_word_reverse_index.jsonl'
        }
    elif args.target_model == "sharegpt4v_ft":
        threshold_dict = {'token': 75, 'object': 19,  'co_occurrence': 6, 'what_word': 203}
        threshold_dict = {'token': 210, 'object': 68,  'co_occurrence': 10, 'what_word': 203}
        reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/dinoobj_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/dinocooccur_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/sharegpt4v_train/ft/reverse_index/whatword_reverse_index.jsonl'
        }
    elif args.target_model == "llava_pt":
        threshold_dict = {'object': 27, 'token': 26, 'co_occurrence': 16, 'what_word': 18695}
        reverse_index_file_dict = {
            'object': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/reverse_index/llamaobj_reverse_index.jsonl',
            'token': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/reverse_index/token_reverse_index.jsonl',
            'co_occurrence': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/reverse_index/llamacooccur_reverse_index.jsonl',
            'what_word': '/mnt/petrelfs/songmingyang/songmingyang/data/llava_train/LLaVA-LCS-558K/dr_algo/reverse_index/whatword_reverse_index.jsonl'
        }
    else:
        raise ValueError("target_model not supported")
    random.seed(args.seed)
    if args.mode =="add":
        sample_data_add(threshold_dict, reverse_index_file_dict, args.input_dataset_file, args.output_file, args.pass_num)
    elif args.mode == "sample":
        sample_data(threshold_dict, reverse_index_file_dict, args.input_dataset_file, args.output_file, args.pass_num)
    elif args.mode == "one":
        assert args.key is not None
        sample_data_one(threshold_dict, reverse_index_file_dict, args.input_dataset_file, args.output_file, args.pass_num,args.key)
    elif args.mode == "compose":
        assert args.compose_list is not None and isinstance(args.compose_list,list)
        assert args.compose_list[0] in ['object','token','co_occurrence','what_word' ]
        sample_data_compose(threshold_dict, reverse_index_file_dict, args.input_dataset_file, args.output_file, args.compose_list, args.pass_num)
    elif args.mode == "compose_alpha":
        assert args.compose_list is not None and isinstance(args.compose_list,list)
        assert args.compose_list[0] in ['object','token','co_occurrence','what_word' ]
        assert args.alpha is not None
        sample_data_compose_alpha(threshold_dict, reverse_index_file_dict, args.input_dataset_file, args.output_file, args.compose_list, args.alpha, args.pass_num)
    else:
        raise ValueError("mode not supported")