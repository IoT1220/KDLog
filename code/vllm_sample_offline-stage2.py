import json
import os
import torch
import copy
from datetime import datetime

from functools import reduce
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer
from argparse import ArgumentParser

from cal_acc import cleanup_response_with_judgement
from prompts import *
from config import *

def get_timestamp() -> str:
    return datetime.now().strftime("%m%d-%H%M")


def make_fewshot_sample(path: str) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    few_shot_examples = [
        [{'role': 'user', 'content': item['user_content']}] + \
        [{'role': 'assistant', 'content': item['assistant_content']}]
        for item in data
    ]
    return reduce(lambda x, y: x + y, few_shot_examples, [])


def add_fewshot_to_system_prompt(system_prompt: str, fewshot: list[dict]) -> str:
    for i in range(0, len(fewshot), 2):
        system_prompt += f"\n\nUser: {fewshot[i]['content']}\n\nAssistant: {fewshot[i+1]['content']}"
    return system_prompt


def make_chat_inputs(tokenizer: PreTrainedTokenizer, system_prompt: str, user_question: str, fewshot: list[dict], max_length=31000) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
    ] + fewshot + [
        {"role": "user", "content": user_question}
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return inputs[-max_length:]




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=DatasetEnum, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--fewshot', type=lambda x: 'y' in x.lower(), default=True)
    parser.add_argument('--fewshot_path', type=str, default=None)
    parser.add_argument('--run_split', default=1, type=int)
    parser.add_argument('--split', default='train', type=str, choices=['train', 'test'])
    args = parser.parse_args()

    print(args)
    # ------------------------------- config starts ------------------------------ #

    if args.fewshot and isinstance(args.fewshot_path, str):
        if not os.path.exists(args.fewshot_path):
            args.fewshot = False
            args.fewshot_path = None

    dataset = args.dataset
    test_data_path = get_data_folder(dataset)
    system_prompt = get_system_prompt(dataset)

    if args.fewshot:
        fewshot_path = get_fewshot_examples_path(dataset)
        if isinstance(args.fewshot_path, str):
            fewshot_path = args.fewshot_path
        fewshot_examples = make_fewshot_sample(fewshot_path)
    else:
        fewshot_examples = []

    model = args.model
    
    # system_prompt = add_fewshot_to_system_prompt(system_prompt, fewshot_examples)

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype='auto', enable_prefix_caching=True, tensor_parallel_size=torch.cuda.device_count(), max_model_len=32000)
    
    test_data_path = f"{test_data_path}/train{args.run_split}+test{args.run_split}/{args.split}.json"

    print(f"sampling path={test_data_path}")

    output_path = args.output
    file_type = 'train' if 'train' in os.path.basename(test_data_path) else 'test'
    if file_type == 'test': args.sample_n = 1

    if output_path is None:
        output_path = os.path.dirname(test_data_path) + '/' + get_timestamp() + '.' + os.path.basename(model) + f'.{file_type}.pred' + f'.sample_n={args.sample_n}' + f'.FS={len(fewshot_examples)}'

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=args.sample_n, max_tokens=768, min_tokens=32, truncate_prompt_tokens=8192)

    # -------------------------------- config done ------------------------------- #

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    inputs = [make_chat_inputs(tokenizer, system_prompt, item['content'], fewshot_examples) for item in test_data]

    outputs = llm.generate(prompt_token_ids=inputs, sampling_params=sampling_params)
    outputs = [[item.text for item in output.outputs] for output in outputs]
    
    results = []

    label_set = list(set(item['label'] for item in test_data))

    with tqdm(total=len(test_data)) as pbar:
        for item, responses in zip(test_data, outputs):

            user_question = item['content']
            label = item['label']


            for response in responses:
                item = copy.deepcopy(item)

                pred, is_correct = cleanup_response_with_judgement(response, label, label_set)
                item['response'] = response
                item['prediction'] = pred
                item['is_correct'] = is_correct
                results.append(item)

            acc = sum([item['is_correct'] for item in results]) / len(results) * 100

            pbar.set_postfix_str(f"Acc: {acc:.2f}%")
            pbar.update(1)


    acc = sum([item['is_correct'] for item in results]) / len(results) * 100
    acc = f"{acc:.2f}"
    output_path += f'.Acc={acc}.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f'model={model}, input={test_data_path}, output={output_path}, acc={acc}\nDone!')
