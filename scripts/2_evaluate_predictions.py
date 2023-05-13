import json
import os
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)


def read_jsonl(path) -> List[Dict]:
    json_lines = []
    with open(path, 'r') as f:
        for i, line in enumerate(f, start=1):
            json_lines.append(json.loads(line))
    print(f'{len(json_lines)} json lines loaded.')
    return json_lines


def parse_non_cot_predictions(path):
    samples = read_jsonl(path)
    predictions = []
    for sample in samples:
        parsed_pred = None
        pred = sample['prediction'].strip()[:3]
        if 'yes' in pred:
            parsed_pred = 'yes'
        elif 'no' in pred:
            parsed_pred = 'no'
        else:
            raise ValueError('Neither yes nor no')
        assert parsed_pred is not None
        predictions.append(parsed_pred)
    return predictions


def parse_cot_zero_shot_predictions(path):
    samples = read_jsonl(path)
    predictions = []
    for i, sample in enumerate(samples, start=1):
        parsed_pred = None
        pred = sample['prediction'].strip()[:3]
        if 'yes' in pred:
            parsed_pred = 'yes'
        elif 'no' in pred:
            parsed_pred = 'no'
        else:
            # try the end of generation
            pred = sample['prediction'].strip()[-6:]
            if 'yes' in pred:
                parsed_pred = 'yes'
            elif 'no' in pred:
                parsed_pred = 'no'
            else:
                if '"yes"' in sample['prediction'].strip() and '"no"' not in sample['prediction'].strip():
                    parsed_pred = 'yes'
                elif '"no"' in sample['prediction'].strip() and '"yes"' not in sample['prediction'].strip():
                    parsed_pred = 'no'
                else:
                    if '\nyes' in sample['prediction'].strip() and '\nno' not in sample['prediction'].strip():
                        parsed_pred = 'yes'
                    elif '\nno' in sample['prediction'].strip() and '\nyes' not in sample['prediction'].strip():
                        parsed_pred = 'no'
                    else:
                        if '"yes,"' in sample['prediction'].strip() and '"no,"' not in sample['prediction'].strip():
                            parsed_pred = 'yes'
                        elif '"no,"' in sample['prediction'].strip() and '"yes,"' not in sample['prediction'].strip():
                            parsed_pred = 'no'
                        else:
                            if '"yes."' in sample['prediction'].strip() and '"no."' not in sample['prediction'].strip():
                                parsed_pred = 'yes'
                            elif '"no."' in sample['prediction'].strip() and '"yes."' not in sample['prediction'].strip():
                                parsed_pred = 'no'
                            else:
                                print(i, sample)
                                raise ValueError('Neither yes nor no')
        assert parsed_pred is not None
        predictions.append(parsed_pred)
    return predictions


def parse_cot_few_shot_predictions(path):
    print(path)
    samples = read_jsonl(path)
    predictions = []
    for i, sample in enumerate(samples, start=1):
        parsed_pred = None
        pred = sample['prediction'].strip()[-6:].lower()
        if 'yes' in pred:
            parsed_pred = 'yes'
        elif 'no' in pred:
            parsed_pred = 'no'
        else:
            print(i, sample)
            raise ValueError('Neither yes nor no')
        assert parsed_pred is not None
        predictions.append(parsed_pred)
    return predictions


if __name__ == '__main__':
    big_bench_dataset = read_jsonl('../data/big_bench_dataset.jsonl')
    do_pigs_fly_dataset = read_jsonl('../data/do_pigs_fly_dataset.jsonl')
    datasets = {
        'big_bench': big_bench_dataset,
        'do_pigs_fly': do_pigs_fly_dataset
    }

    predictions = {
        'big_bench': {},
        'do_pigs_fly': {}
    }

    for dataset_name, dataset in datasets.items():
        path = os.path.join('../data/predictions', f'open_ai_text-davinci-003-{dataset_name}-zero_shot_predictions.jsonl')
        preds = parse_non_cot_predictions(path)
        predictions[dataset_name]['zero-shot'] = preds

        path = os.path.join('../data/predictions', f'open_ai_text-davinci-003-{dataset_name}-few_shot_predictions.jsonl')
        preds = parse_non_cot_predictions(path)
        predictions[dataset_name]['few-shot'] = preds

        # path = os.path.join('../data/predictions', f'open_ai_text-davinci-003-{dataset_name}-zero_shot_cot_predictions.jsonl')
        # preds = parse_cot_zero_shot_predictions(path)
        # predictions[dataset_name]['zero_shot_cot'] = preds

        path = os.path.join('../data/predictions', f'open_ai_text-davinci-003-{dataset_name}-few_shot_cot_predictions.jsonl')
        preds = parse_cot_few_shot_predictions(path)
        predictions[dataset_name]['chain-of-thought'] = preds

    big_bench_scores = {}
    answers = [sample['answer'] for sample in datasets['big_bench']]
    for mode, preds in predictions['big_bench'].items():
        # print(mode)
        # print(classification_report(answers, preds))
        prec, rec, f1, sup = precision_recall_fscore_support(answers, preds, average='macro')
        acc = accuracy_score(answers, preds)
        big_bench_scores[mode] = {'precision': prec, 'recall': rec, 'f1-score': f1, 'accuracy': acc}
    df_big_bench = pd.DataFrame.from_dict(big_bench_scores, orient='index').round(decimals=2)

    do_pigs_fly_scores = {}
    answers = [sample['answer'] for sample in datasets['do_pigs_fly']]
    for mode, preds in predictions['do_pigs_fly'].items():
        # print(mode)
        # print(classification_report(answers, preds))
        prec, rec, f1, sup = precision_recall_fscore_support(answers, preds, average='macro')
        acc = accuracy_score(answers, preds)
        do_pigs_fly_scores[mode] = {'precision': prec, 'recall': rec, 'f1-score': f1, 'accuracy': acc}
    df_do_pigs_fly = pd.DataFrame.from_dict(do_pigs_fly_scores, orient='index').round(decimals=2)

    print(df_big_bench)
    print(df_do_pigs_fly)
    print(df_big_bench.to_latex())
    print(df_do_pigs_fly.to_latex())