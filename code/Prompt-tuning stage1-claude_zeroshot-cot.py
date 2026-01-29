import time
from openai import OpenAI
from collections import Counter
from loguru import logger
import json
import os

os.makedirs('output-stage1', exist_ok=True)

api_key = "sk-YzgzxE95qx8bQAhY45C4BdAdE44245D5Ac2b7e1b981cC91a"
base_url = "https://api.132006.xyz/v1"
client = OpenAI(api_key=api_key, base_url=base_url)


label2name = {1: 'Power Supply Fault', 2: 'Fan Fault', 3: 'Optics Module Fault', 4: 'Port Failure', 6: 'CRC Error', 7: 'STP Fault', 8: 'BFD Down', 9: 'LACP Flapping', 10: 'OSPF Neighbor Flapping'}

label2details = {1: 'Power Supply Fault', 2: 'Fan Fault', 3: 'Optics Module Fault', 4: 'Port Failure',
                 6: 'CRC Error (Cyclic Redundancy Check)', 7: 'STP Fault (Spanning Tree Protocol)',
                 8: 'BFD Down (Bidirectional Forwarding Detection)', 9: 'LACP Flapping (Link Aggregation Control Protocol)',
                 10: 'OSPF Neighbor Flapping (Open Shortest Path First)'}


def load_data(filename):

    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'Loaded {len(data)} records in total')

    # # collect 1 record for each class
    # cleaned_data = []
    # label_counter = Counter()
    # for record in data:
    #     label = record['label']
    #     if label_counter[label] < 1:
    #         cleaned_data.append(record)
    #         label_counter[label] += 1

    # logger.info(f'Loaded {len(cleaned_data)} records after filtering')
    # return cleaned_data

    # desired_caseid = [9, 18, 27, 36, 45, 50]

    desired_caseid = list(range(50))
    logger.info(f'Loading records with caseid: {desired_caseid}')

    cleaned_data = []
    for record in data:
        if record['caseid'] in desired_caseid:
            cleaned_data.append(record)

    # sort the data by caseid
    cleaned_data = sorted(cleaned_data, key=lambda x: x['caseid'])

    logger.info(f'Loaded {len(cleaned_data)} records after filtering')

    return cleaned_data


def analyze_label_distribution(data):
    labels = [record['label'] for record in data]
    label_counter = Counter(labels)
    logger.info(f'Label distribution: {label_counter}')


def truncate_log(log, max_len=1e5):
    if len(log) > max_len:
        logger.debug(f'Log exceeds the maximum length, truncating it to {max_len}')
        return log[:max_len]
    return log


def get_completion(messages, model="claude-3-5-sonnet-20240620"):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content


def analyze_log(log):

    system_prompt = f'''
    You will work as a text classification model to classify the operation error logs from the system to the defined categories. The categories are: {", ".join(label2details.values())}.
    '''.strip()

    user_prompt_round1 = f"""
    The operation error log is:
    {log}

    You are now acting as a human labeler who writes key analysis points for the error. Comparing it to all the possible error types (e.g. 1. Compared to ..., this error log ... 2. ...). There is no need to conclude the error type in the analysis since we will use the analysis to classify the error type in next steps.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_round1}
    ]
    return get_completion(messages)


def classify_log(log, analysis):

    system_prompt = f'''
    You will work as a text classification model to classify the operation error logs from the system to the defined categories. The categories are: {", ".join(label2details.values())}.
    '''.strip()
    user_prompt_round1 = f"""
    The operation error log is:
    {log}

    You are now acting as a human labeler who writes key analysis points for the error. Comparing it to all the possible error types (e.g. 1. Compared to ..., this error log ... 2. ...). There is no need to conclude the error type in the analysis since we will use the analysis to classify the error type in next steps.
    """.strip()

    user_prompt_round2 = f"""
    Based on the analysis you've just provided, please classify the error type of the log. You can choose from the following categories: {", ".join(label2details.values())}. Please directly provide the error type without any additional analysis, and the provided error type should be absolutely consistent with the label names we've defined.
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_round1},
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


def process_records(data):

    results = []

    for i, record in enumerate(data):

        content = truncate_log(record['content'])
        analysis = analyze_log(content)

        classification, chat_history = classify_log(content, analysis)

        max_retry = 3
        matched_label = None

        while max_retry > 0:
            matched_label = match_response(classification, label2name)
            if matched_label is not None:
                break
            logger.warning(f'Failed to match the response for case {record["caseid"]} part {record["part"]}, retrying for {3-max_retry+1} times')
            classification, chat_history = classify_log(content, analysis)
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

        results.append(result)
        if matched_label != label2name[record['label']]:
            logger.warning(f'Processed case {record["caseid"]} part {record["part"]}: True Label: {result["true_label"]}, Matched Label: {result["matched_label"]}')
        else:
            logger.success(f'Processed case {record["caseid"]} part {record["part"]}: True Label: {result["true_label"]}, Matched Label: {result["matched_label"]}')

    return results


def save_results(results, timestamp, filename='output-stage1/claude_results.json'):

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


def main():

    timestamp = time.strftime('%Y%m%d%H%M%S')
    logger.add(f'output-stage1/claude_{timestamp}.log')

    data = load_data('data/output.json')
    analyze_label_distribution(data)

    start_time = time.time()
    results = process_records(data)
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
