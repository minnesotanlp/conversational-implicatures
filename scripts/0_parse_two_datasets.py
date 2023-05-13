import json
import csv


def parse_datasets(big_bench_dataset_path = '../data/BIG-bench_implicatures_2957b2d.json',
                   do_pigs_fly_dataset_path = '../data/do-pigs-fly_test_conversational_implicatures_ef8f4fd.csv'):
    big_bench_dataset = []
    with open(big_bench_dataset_path, 'r') as f:
        big_bench_dataset_raw = json.load(f)['examples']
        for example in big_bench_dataset_raw:
            # {
            #     "input": "Speaker 1: 'You're both comin', right?' Speaker 2: 'We're already here.'",
            #     "target_scores": {
            #         "yes": 1.0,
            #         "no": 0.0
            #     }
            # }
            _input = example['input']
            question, response = example['input'].split('Speaker 2:')
            question = question.replace('Speaker 1:', '').strip()
            response = response.strip()
            is_yes = example['target_scores']['yes'] == 1.0
            is_no = example['target_scores']['no'] == 1.0
            answer = None
            if is_yes:
                answer = 'yes'
            elif is_no:
                answer = 'no'
            else:
                raise ValueError('neither fully yes nor no')

            if question[0] == '\'' and question[-1] == '\'':
                question = question[1:-1]
            elif question[0] == '"' and question[-1] == '"':
                question = question[1:-1]

            if response[0] == '\'' and response[-1] == '\'':
                response = response[1:-1]
            elif response[0] == '"' and response[-1] == '"':
                response = response[1:-1]

            big_bench_dataset.append({'question': question, 'response': response, 'answer': answer})

    print(len(big_bench_dataset))

    do_pigs_fly_dataset = []
    with open(do_pigs_fly_dataset_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for question, response, answer in csv_reader:
            is_yes = answer.lower()[:len('yes')] == 'yes'
            is_no = answer.lower()[:len('no')] == 'no'
            if is_yes:
                answer = 'yes'
            elif is_no:
                answer = 'no'
            else:
                raise ValueError('neither fully yes nor no')
            do_pigs_fly_dataset.append({'question': question.strip(), 'response': response.strip(), 'answer': answer.strip()})

    print(len(do_pigs_fly_dataset))

    with open('../data/big_bench_dataset.jsonl', 'w') as f:
        for sample in big_bench_dataset:
            f.write(f'{json.dumps(sample, indent=None)}\n')

    with open('../data/do_pigs_fly_dataset.jsonl', 'w') as f:
        for sample in do_pigs_fly_dataset:
            f.write(f'{json.dumps(sample, indent=None)}\n')

    return big_bench_dataset, do_pigs_fly_dataset


big_bench_dataset, do_pigs_fly_dataset = parse_datasets()