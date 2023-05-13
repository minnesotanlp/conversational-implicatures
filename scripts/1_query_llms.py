import json
import os
import time
from typing import Any, Dict, List

import yaml
from rich.progress import track
from talkative_llm.llm import LLMCaller, get_supported_llm


def chunk_with_size_n(list_of_items: List[Any], chunk_size: int) -> List[Any]:
    for i in range(0, len(list_of_items), chunk_size):
        yield list_of_items[i:i + chunk_size]


def load_llm(framework: str, model: str) -> Dict:
    config_name = f'{framework}_{model}.yaml'.replace('/', '_')
    config_path = os.path.join('../llm_configs', config_name)
    return load_llm_from_yaml_config(config_path)


def load_llm_from_yaml_config(config_path: str) -> LLMCaller:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return get_supported_llm(config)


def read_jsonl(path) -> List[Dict]:
    json_lines = []
    with open(path, 'r') as f:
        for line in f:
            json_lines.append(json.loads(line))
    print(f'{len(json_lines)} json lines loaded.')
    return json_lines


def write_predictions(predictions, path):
    with open(path, 'w') as f:
        for pred in predictions:
            f.write(f'{json.dumps(pred, indent=None)}\n')


def complete_by_llm(llm, dataset, template):
    prompts = []
    for sample in dataset:
        prompt = template.format(sample['question'], sample['response'])
        prompts.append(prompt)

    generations = []
    for prompt_batch in track(chunk_with_size_n(prompts, 20), description="Generating..."):
        results = llm.generate(prompt_batch)
        generations.extend(results)
        time.sleep(10)

    if len(generations) != len(dataset):
        print(f'len of generations does not match the original dataset: {len(generations)} != {len(dataset)}')

    predictions = []
    for gen in generations:
        if gen['finish_reason'] != 'stop':
            predictions.append(None)
        else:
            pred = gen['generation'].strip().lower()
            predictions.append({'prediction': pred})
    return predictions


if __name__ == '__main__':
    big_bench_dataset = read_jsonl('../data/big_bench_dataset.jsonl')
    do_pigs_fly_dataset = read_jsonl('../data/do_pigs_fly_dataset.jsonl')
    datasets = {
        'big_bench': big_bench_dataset,
        'do_pigs_fly': do_pigs_fly_dataset
    }

    zero_shot_prompt_template = ('Esther and Juan are engaging in a conversation.\n'
                                 'Esther asked "{}" and Juan responded "{}"\n'
                                 'What does Juan mean by his response? Answer by either "yes" or "no".\n')

    few_shot_prompt_template = ('The following examples are coherent sentences:\n'
                                'Esther asked "Have you found him yet?" and Juan responded "They\'re still looking", which means no.\n'
                                'Esther asked "Are you having fun?" and Juan responded "Is the pope Catholic?", which means yes.\n'
                                'Similarly, please, finish the following sentence by either "yes" or "no":\n'
                                'Esther asked "{}" and Juan responded "{}", which means\n')

    zero_shot_cot_prompt_template = ('Esther and Juan are engaging in a conversation.\n'
                                     'Esther asked "{}" and Juan responded "{}"\n'
                                     'What does Juan mean by his response? Answer by either "yes" or "no".\n'
                                     'Let\'s think step by step.\n')

    few_shot_cot_prompt_template = ('Esther and Juan are engaging in a conversation.\n'
                                    'Esther asked "Juan, are you going to Paul\'s party?" and Juan responded "I have to work late."\n'
                                    'What does Juan mean by his response? Answer by either "yes" or "no".\n\n'
                                    'Esther wants to know an answer to her question: "Have you found him yet?"\n'
                                    'Juan responds: "They\'re still looking"\n'
                                    'Juan\'s response, in literal sense, does not count as a direct answer to the question.\n'
                                    'On its face, Juan appears to be talking about something irrelevant to the question.\n'
                                    'Yet Esther has no reason to believe that Juan is opting out of the operation of the cooperative principle '
                                    'which assumes that participants in a conversation cooperate with each other and attempt to be '
                                    'truthful, informative, relevant, and clear in order to facilitate successful communication.\n'
                                    'Juan must therefore intend for Esther to infer an answer from "what was said" and background knowledge.\n'
                                    'What would be the relevant background knowledge in this situation?\n'
                                    'It is the fact that work-related responsibilities typically take precedence over temporally co-located social events.\n'
                                    'Juan must therefore intend for Esther to infer that he will not attend the party due to him having to work late.\n'
                                    'Thus, Juan means "no" from his response.\n'
                                    'Answer: No.\n\n'
                                    'Esther and Juan are engaging in a conversation.\n'
                                    'Esther asked "Are you having fun?" and Juan responded "Is the pope Catholic?"\n'
                                    'What does Juan mean by his response? Answer by either "yes" or "no".\n\n'
                                    'Esther wants to know an answer to her question: "Are you having fun?"\n'
                                    'Juan responds: "Is the pope Catholic?"\n'
                                    'Juan\'s response, in literal sense, does not count as a direct answer to the question.\n'
                                    'On its face, Juan appears to be talking about something irrelevant to the question.\n'
                                    'Yet Esther has no reason to believe that Juan is opting out of the operation of the cooperative principle '
                                    'which assumes that participants in a conversation cooperate with each other and attempt to be '
                                    'truthful, informative, relevant, and clear in order to facilitate successful communication.\n'
                                    'Juan must therefore intend for Esther to infer an answer from "what was said" and background knowledge.\n'
                                    'What would be the relevant background knowledge in this situation?\n'
                                    'It is the fact that the pope is the head of the Catholic Church so it is trivially obvious that he is Catholic. '
                                    'And this a common way of rhetorically responding to a question to which the answer is an emphatic yes.\n'
                                    'Juan must therefore intend for Esther to infer that he is indeed having fun.\n'
                                    'Thus, Juan means "yes" from his response.\n'
                                    'Answer: Yes.\n\n'
                                    'Esther and Juan are engaging in a conversation.\n'
                                    'Esther asked "{}" and Juan responded "{}"\n'
                                    'What does Juan mean by his response? Answer by either "yes" or "no".\n')

    open_ai_model = load_llm('openai', 'text-davinci-003')

    models = {
        'open_ai_text-davinci-003': open_ai_model
    }

    for model_name, model in models.items():
        for dataset_name, dataset in datasets.items():
            # Non-CoT
            zero_shot_predictions = complete_by_llm(model, dataset, zero_shot_prompt_template)
            write_predictions(zero_shot_predictions, f'../data/predictions/{model_name}-{dataset_name}-zero_shot_predictions.jsonl')
            few_shot_predictions = complete_by_llm(model, dataset, few_shot_prompt_template)
            write_predictions(few_shot_predictions, f'../data/predictions/{model_name}-{dataset_name}-few_shot_predictions.jsonl')

            # CoT
            zero_shot_cot_predictions = complete_by_llm(model, dataset, zero_shot_cot_prompt_template)
            write_predictions(zero_shot_cot_predictions, f'../data/predictions/{model_name}-{dataset_name}-zero_shot_cot_predictions.jsonl')
            few_shot_cot_predictions = complete_by_llm(model, dataset, few_shot_cot_prompt_template)
            write_predictions(few_shot_cot_predictions, f'../data/predictions/{model_name}-{dataset_name}-few_shot_cot_predictions.jsonl')