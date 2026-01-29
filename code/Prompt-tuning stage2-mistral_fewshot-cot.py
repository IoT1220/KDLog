import time
from openai import OpenAI
from collections import Counter
from loguru import logger
import json
import os
from tqdm import tqdm

os.makedirs('output-stage2-fewshot-cot', exist_ok=True)

# CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --dtype auto --api-key token-abc123s --port 18889 --gpu-memory-utilization 0.99
client = OpenAI(
    base_url="http://localhost:18889/v1",
    api_key="token-abc123s",
)


label2name = {1: 'Power Supply Fault', 2: 'Fan Fault', 3: 'Optics Module Fault', 4: 'Port Failure', 6: 'CRC Error', 7: 'STP Fault', 8: 'BFD Down', 9: 'LACP Flapping', 10: 'OSPF Neighbor Flapping'}

label2details = {1: 'Power Supply Fault', 2: 'Fan Fault', 3: 'Optics Module Fault', 4: 'Port Failure',
                 6: 'CRC Error (Cyclic Redundancy Check)', 7: 'STP Fault (Spanning Tree Protocol)',
                 8: 'BFD Down (Bidirectional Forwarding Detection)', 9: 'LACP Flapping (Link Aggregation Control Protocol)',
                 10: 'OSPF Neighbor Flapping (Open Shortest Path First)'}


def load_data(filename):

    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'Loaded {len(data)} records in total')

    un_desired_caseid = list(range(50))
    # now we will be testing examples from 50 to the end
    # logger.info(f'Loading records with caseid: {desired_caseid}')

    cleaned_data = []
    for record in data:
        if record['caseid'] not in un_desired_caseid:
            cleaned_data.append(record)

    # on keep those for Optics Module Fault, Port Failure, BFD Down and LACP Flapping
    # cleaned_data = [record for record in cleaned_data if record['label'] in [3, 4, 8, 9]]
    # cleaned_data = [record for record in cleaned_data if record['label'] in [8, 9]]

    # sort the data by caseid
    cleaned_data = sorted(cleaned_data, key=lambda x: x['caseid'])

    logger.info(f'Loaded {len(cleaned_data)} records after filtering')

    return cleaned_data


def analyze_label_distribution(data):
    
    labels = [record['label'] for record in data]
    label_counter = Counter(labels)
    logger.info(f'Label distribution: {label_counter}')


def truncate_log(log, max_len=1e5, verbose=False):
    
    if len(log) > max_len:
        if verbose:
            logger.debug(f'Log exceeds the maximum length, truncating it to {max_len}')
        return log[:max_len]
    
    return log


def get_completion(messages, model="mistralai/Mistral-7B-Instruct-v0.3"):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content


def analyze_log(log, few_shot_examples):

    few_shot_content = ""
    for i, example in enumerate(few_shot_examples):
        
        few_shot_content += f'''\n\nSystem Log of Example {i}: {truncate_log(example['content'], max_len=256)}...\nLabel of Example {i}: {example['matched_label']}'''

    system_prompt = f'''
    You will work as a text classification model to classify the operation error logs from the system to the defined categories. The categories are: \n{", ".join(label2details.values())}\nHre are examples of the error logs and their corresponding error types, together with the analysis points. You can use these examples to help you classify the error types of the operation error logs.
    {few_shot_content}
    Please remember to relate to these examples when analyzing and classifying the error types of the operation error logs.
    '''.strip()

    user_prompt_round1 = f"""
    The operation error log to analysis now is:
    {log}

    # You are now acting as a human labeler who writes key analysis points for the error. Comparing it to all the possible error types and the examples (e.g. 1. Compared to ..., this error log ... 2. ...). There is no need to conclude the error type in the analysis since we will use the analysis to classify the error type in next steps. You should try to analyze the error based on both your understanding of the error types and the examples provided.
    """.strip()
    

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_round1}
    ]

    return get_completion(messages), messages


def classify_log(log, analysis, chat_history):

    user_prompt_round2 = f"""
    The operation error log to classify now is:
    {log}
    
    Based on the analysis you've just provided, please classify the error type of the log. You can choose from the following categories: {", ".join(label2details.values())}. Please directly provide the error type without any additional analysis, and the provided error type should be absolutely consistent with the label names we've defined.
    """.strip()

    messages = chat_history + [
        {"role": "assistant", "content": analysis},
        {"role": "user", "content": user_prompt_round2}
    ]

    return get_completion(messages), messages


def match_response(response, label2name):

    response_words = response.lower().split()
    all_labels = list(label2name.values())
    max_common_words = 0
    matched_label = None

    for label in all_labels:

        label_words = label.lower().split()
        common_words = len(set(response_words).intersection(label_words))

        if common_words > max_common_words:
            max_common_words = common_words
            matched_label = label

    if matched_label is None:
        logger.warning(f'No matched label found for response: {response}')

    return matched_label


def process_records(data, few_shot_examples):

    results = []

    for i, record in enumerate(tqdm(data, ncols=100, desc='Processing Records')):

        try:
            content = truncate_log(record['content'])
            analysis, chat_history = analyze_log(content, few_shot_examples)
            classification, chat_history = classify_log(content, analysis, chat_history)
        except Exception as e:
            logger.error(f'Error processing case {record["caseid"]} part {record["part"]}: {e}')
            content = truncate_log(record['content'], max_len=4096)
            analysis, chat_history = analyze_log(content, few_shot_examples)
            classification, chat_history = classify_log(content, analysis, chat_history)
            

        max_retry = 3
        matched_label = None

        while max_retry > 0:
            matched_label = match_response(classification, label2name)
            if matched_label is not None:
                break
            logger.warning(f'Failed to match the response for case {record["caseid"]} part {record["part"]}, retrying for {3-max_retry+1} times')
            classification, chat_history = classify_log(content, analysis, chat_history)
            max_retry -= 1



        chat_history.append({"role": "assistant", "content": classification})

        result = {
            'caseid': record['caseid'],
            'part': record['part'],
            'content': content,
            'matched_label': matched_label,
            'true_label': label2name[record['label']],
            'correct': matched_label == label2name[record['label']],
            'chat_history': chat_history
        }

        # logger.debug(f'Result for case {record["caseid"]} part {record["part"]}: {result["chat_history"]}')

        results.append(result)
        if matched_label != label2name[record['label']]:
            logger.warning(f'Processed case {record["caseid"]} part {record["part"]}: True Label: {result["true_label"]}, Matched Label: {result["matched_label"]}')
        else:
            logger.success(f'Processed case {record["caseid"]} part {record["part"]}: True Label: {result["true_label"]}, Matched Label: {result["matched_label"]}')

    return results


def save_results(results, timestamp, filename='output-stage2-fewshot-cot/mistral_results.json'):

    filename = filename.replace('.json', f'_{timestamp}.json')

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f'Results saved to {filename}')


def analyze_correctness(results):

    correct = 0

    for result in results:
        if result['true_label'] == result['matched_label']:
            correct += 1

    accuracy = correct / len(results)
    logger.info(f'\nAccuracy: {accuracy*100:.2f}%')

    # get classification report
    from sklearn.metrics import classification_report
    y_true = [result['true_label'] for result in results]
    y_pred = [result['matched_label'] for result in results]

    all_labels = list(label2name.values())

    report = classification_report(y_true, y_pred, target_names=all_labels, labels=all_labels, digits=4, zero_division=0)
    logger.info(f'\nClassification Report:\n{report}')

    # get the confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    logger.info(f'\nConfusion Matrix:\n{cm}', serialize=True)


def load_few_shot_examples(filename='output-stage1/claude_results_20240719173356.json'):

    # we randomly select one for each class
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)

    # shuffle the data
    # import random
    # random.shuffle(data, random=lambda: 0.1)

    few_shot_examples = []
    desired_example_num = {"Power Supply Fault": 1, "Fan Fault": 1, "Optics Module Fault": 1, "Port Failure": 1, "CRC Error": 1, "STP Fault": 1, "BFD Down": 1, "LACP Flapping": 1, "OSPF Neighbor Flapping": 1}

    # for record in data:
    #     if desired_example_num.get(record['matched_label'], 0) > 0:
    #         few_shot_examples.append(record)
    #         desired_example_num[record['matched_label']] -= 1

    # desired_example_num = {"Power Supply Fault": 1, "Fan Fault": 1, "Optics Module Fault": 2, "Port Failure": 2, "CRC Error": 1, "STP Fault": 1, "BFD Down": 2, "LACP Flapping": 2, "OSPF Neighbor Flapping": 2}

    for record in data:
        if desired_example_num.get(record['matched_label'], 0) > 0:
            few_shot_examples.append(record)
            desired_example_num[record['matched_label']] -= 1

    # sort the examples by label in the label2name order
    few_shot_examples = sorted(few_shot_examples, key=lambda x: list(label2name.keys())[list(label2name.values()).index(x['matched_label'])])

    logger.info(f'Loaded {len(few_shot_examples)} few-shot examples')

    return few_shot_examples


def main():

    timestamp = time.strftime('%Y%m%d%H%M%S')
    logger.add(f'output-stage2-fewshot-cot/mistral_{timestamp}.log')

    data = load_data('data/output.json')
    analyze_label_distribution(data)

    few_shot_examples = load_few_shot_examples()

    start_time = time.time()
    results = process_records(data, few_shot_examples)
    end_time = time.time()
    logger.info(f'Processing time: {end_time-start_time:.2f} seconds, average: {(end_time-start_time)/len(data):.2f} seconds per record')

    analyze_correctness(results)
    save_results(results, timestamp)


def test(output):

    with open(output, encoding='utf-8') as f:
        results = json.load(f)

    analyze_correctness(results)


if __name__ == "__main__":
    main()
