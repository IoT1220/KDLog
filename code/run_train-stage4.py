import yaml
import json
import os
from argparse import ArgumentParser
from datetime import datetime

def get_timestamp() -> str:
    return datetime.now().strftime("%m%d")

if __name__ == '__main__':
    with open('data/dataset_info.json', 'r') as f:
        dataset_keys = list(json.load(f).keys())


    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=dataset_keys)
    parser.add_argument('--train_method', type=str, choices=['orpo', 'dpo', 'ipo'])
    parser.add_argument('--output', type=str)
    parser.add_argument('--base_config', type=str, default='orpo.config.yaml')
    parser.add_argument('--mode', type=str, choices=['dpo', 'sft'], default='dpo')
    parser.add_argument('--ckpt', type=str, default="/home/xuting/Mistral-7B-Instruct-v0.3")

    args = parser.parse_args()
    print(args)

    base_dir = os.path.basename(args.output)
    method = args.mode + '-' + args.train_method
    output_merged = args.output + '/' + f"{base_dir}-{method}-merged"
    print(f"the final merged ckpt={output_merged}")
    

    if args.mode == 'sft':
        args.base_config = 'sft.config.yaml'
        # modify the dataset
        with open('data/dataset_info.json', 'r') as f:
            dataset_json_config = json.load(f)
        
        if not args.dataset.endswith('sft') and (temp := args.dataset + '_' + 'sft') not in dataset_json_config:
            d_path = dataset_json_config[args.dataset]['file_name']
            with open(d_path, 'r') as f:
                data = json.load(f)
            new_data = []
            for item in data:
                item = {
                    'instruction': item['instruction'],
                    'input': item['input'],
                    'output': item['chosen'],
                    "system": "",
                    "history": [],
                }
                new_data.append(item)
            d_path = d_path + '_sft.json' 
            with open(d_path, 'w') as f:
                json.dump(new_data, f)

            dataset_json_config[temp] = {
                "file_name": d_path,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history"
                }
            }
            with open('data/dataset_info.json', 'w') as f:
                json.dump(dataset_json_config, f, indent=4)

            args.dataset = temp
        
        elif (temp := args.dataset + '_' + 'sft') in dataset_json_config:
            args.dataset = temp



    if args.train_method == 'dpo':
        args.train_method = 'sigmoid'

    with open(args.base_config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    

    if args.mode == 'dpo':
        config['pref_loss'] = args.train_method


    config['dataset'] = args.dataset
    config['output_dir'] = args.output
    config["model_name_or_path"] = args.ckpt

    with open('temp.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print('new yaml generate done!')

    os.system(
        'llamafactory-cli train temp.yaml'
    )

    # merge model
    with open('temp.lora.merge.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    

    config['adapter_name_or_path'] = args.output
    config['export_dir'] = output_merged

    with open('temp.lora.merge.yaml', 'w') as f:
        yaml.dump(config, f)

    os.system('llamafactory-cli export temp.lora.merge.yaml')

    print(f"merged ckpt path=\n{config['export_dir']}")
