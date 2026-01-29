import json
import os
import itertools
from argparse import ArgumentParser

from config import *
import copy

"""
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "chosen": "chosen answer (required)",
    "rejected": "rejected answer (required)"
  }
]
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--dataset', type=DatasetEnum, default=DatasetEnum.aliyun)
    args = parser.parse_args()

    print(args)

    system_prompt = get_system_prompt(args.dataset)

    # concat original data
    data = []

    for file in os.listdir(args.input_folder):
        if os.path.isfile(temp := os.path.join(args.input_folder, file)) and 'train.pred' in file:
            with open(temp, 'r') as f:
                data += json.load(f)
    
    print('total data size = ', len(data))

    # temporary data
    
    if 'reward_score' in data[0]:
        data_dict = {}
        scores = {}

        for item in data:
            caseid = item['caseid']
            if caseid not in data_dict:
                data_dict[caseid] = {
                    'chosen': [],
                    'rejected': [],
                }
                scores[caseid] = {
                    'chosen': -1,
                    'rejected': 1000,
                }
            
            s = item['reward_score']
            if s > scores[caseid]['chosen']:
                scores[caseid]['chosen'] = s
                data_dict[caseid]['chosen'] = [item]
            
            elif s == scores[caseid]['chosen']:
                data_dict[caseid]['chosen'].append(item)

            elif s < scores[caseid]['rejected']:
                scores[caseid]['rejected'] = s
                data_dict[caseid]['rejected'] = [item]

            elif s == scores[caseid]['rejected']:
                data_dict[caseid]['rejected'].append(item)
                


    else:
        data_dict = {}
        for item in data:
            caseid = item['caseid']
            if caseid not in data_dict:
                data_dict[caseid] = {
                    'chosen': [],
                    'rejected': [],
                }
            
            key = 'chosen' if item['is_correct'] else 'rejected'
            data_dict[caseid][key].append(item)



    
    # preference data
    preference_data = []
    caseid_preference_data = []

    for case_id, data in data_dict.items():
        if len(data['chosen']) > 0 and len(data['rejected']) > 0:
            limit_num = 2000
            for chosen, rejected in itertools.product(data['chosen'], data['rejected']):
                preference_data.append({
                    'instruction': system_prompt,
                    'input': chosen['content'],
                    'chosen': chosen['response'],
                    'rejected': rejected['response'],
                })
                temp = copy.deepcopy(preference_data[-1])
                temp['caseid'] = case_id
                caseid_preference_data.append(temp)

                limit_num -= 1
                if limit_num < 0: break

    

    print('preference data size = ', len(preference_data))


    output_path = os.path.join(args.input_folder, 'preference_data.train.json')
    caseid_output_path = output_path.replace('.json', '.caseid.json')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(preference_data, f, indent=4)
    
    with open(caseid_output_path, 'w') as f:
        json.dump(caseid_preference_data, f, indent=4)
            
    print('preference data save to', output_path)
