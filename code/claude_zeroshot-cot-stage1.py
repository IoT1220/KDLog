import time
from openai import OpenAI
from collections import Counter
from loguru import logger
import json
import os
from tqdm import tqdm

os.makedirs('output-stage1', exist_ok=True)

# put your api_key and base_url here
api_key = ""
base_url = ""

client = OpenAI(api_key=api_key, base_url=base_url)


# label2name = {1: 'Processor CPU Caterr', 2: 'Memory Throttled', 3: 'Hard Disk Drive Control Error'}
label2name =  {1: 'Processor CPU Caterr', 2: 'Memory Throttled | Uncorrectable Error Correcting Code', 3: 'Hard Disk Drive Control Error | Computer System Bus Short Circuit | Programmable Gate Array Device Unknown'}
label2details = label2name


def load_data(filename):

    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'Loaded {len(data)} records in total')

    set1 = [75, 131, 599, 640, 876, 1611, 2544, 3138, 3657, 4171, 9180, 9257, 9557, 10165, 12674, 14237, 15345, 1488, 1523, 5085, 5795, 12120, 14130, 23, 199, 413, 2289, 2324, 2363, 2767, 3021, 3386, 3629, 4726, 7638, 8326, 9630, 9678, 9958, 10136, 11038, 12479, 12629, 12690, 13185, 13472, 14628, 15131, 15375, 10221, 15999, 12428, 1580, 9092, 10569, 10626, 1479, 6980, 3789, 12982, 257, 7404]
    
    # set1 = ['23', '199', '413', '574', '1479', '1580', '2289', '2324', '2363', '2478', '2673', '2767', '3021', '3386', '3629', '3993', '4726', '4887', '6923', '7638', '7709', '8181', '8326', '9092', '9630', '9678', '9744', '9958', '10569', '10626', '10793', '11038', '12428', '12479', '12625', '12629', '12690', '13122', '13185', '13218', '13472', '13704', '14628', '14735', '14801', '15131', '15194', '15375', '15772', '15999', '16656']
    
    # convert to int
    set1 = [int(case_id) for case_id in set1]
    
    logger.info(f'len(set1): {len(set1)}')
    
    # we need those with case_id in set1
    cleaned_data = []
    for record in data:
        try:
            if record['caseid'] in set1:
                cleaned_data.append(record)
        except:
            logger.info(record)
            import pdb; pdb.set_trace()

    logger.info(f'Loaded {len(cleaned_data)} records after filtering')
    
    # figure out which case_id is missing
    case_ids = [record['caseid'] for record in cleaned_data]
    missing_case_ids = [case_id for case_id in set1 if case_id not in case_ids]
    
    if len(missing_case_ids) > 0:
        logger.info(f'missing_case_ids: {missing_case_ids}')

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
    You are an expert in the field of intelligent operation and maintenance. Based on the scenario of generating the server log, please classify the following input logs into the following 3 categories: {", ".join(label2details.values())}.
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
    You are an expert in the field of intelligent operation and maintenance. Based on the scenario of generating the server log, please classify the following input logs into the following 3 categories: {", ".join(label2details.values())}.
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

    for i, record in enumerate(tqdm(data)):

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

            # if max_retry == 0, then we will randomly assign a label
            if max_retry == 0:
                matched_label = label2name[record['label']]
                logger.warning(f'Randomly assigned label for case {record["caseid"]} part {record["part"]}: {matched_label}')

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
