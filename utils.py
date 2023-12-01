import math
import json
from warnings import warn
from pathlib import Path
from random import shuffle
from pprint import pformat
from openai import OpenAI
from config import api_key, seed

client = OpenAI(api_key=api_key)
results_path = Path.cwd() / "results"
if not results_path.exists():
    results_path.mkdir()

def load_dataset(dataset_file):
    dataset = json.load(open(dataset_file, "r"))
    tag_counts = {}
    for sample in dataset:
        for tag in sample["tags"]:
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1
    return dataset, tag_counts

def save_dataset(dataset, dataset_file):
    json.dump(dataset, open(dataset_file, "w"), indent=2)

def check_tags(sample, tags, comb):
    if len(tags) == 0:
        return True
    if comb == "AND":
        return all([tag in sample["tags"] for tag in tags])
    elif comb == "OR":
        return any([tag in sample["tags"] for tag in tags])
    else:
        raise f"Invalid combination: {comb}, must be \"AND\" or \"OR\""

default_methods = ["positive", "negative", None]
def create_dataset(dataset, tags_true, tags_false=[], tags_common=[],
                   true_comb="AND", false_comb="OR", default_method="positive"):

    tags_true, tags_false, tags_common = set(tags_true), set(tags_false), set(tags_common)
    if len(tags_true & tags_false) > 0:
        warn(f"Tags in both tags_true and tags_false: {tags_true & tags_false}")
    if len(tags_true) == 0:
        warn("tags_true is empty")

    positives, negatives = [], []
    for sample in dataset:
        true_status = check_tags(sample, tags_true, true_comb)
        false_status = check_tags(sample, tags_false, false_comb)
        common = all([tag in sample["tags"] for tag in tags_common]) if len(tags_common) > 0 else True
        positive = true_status and common
        negative = false_status and common
        if positive and negative:
            if default_method is None:
                continue
            elif default_method == "positive":
                negative = False
            elif default_method == "negative":
                positive = False
            else:
                warn(f"Invalid default method: {default_method}, must be in {default_methods}")
                continue
        if positive:
            sample["label"] = True 
            positives.append(sample)
        elif negative:
            sample["label"] = False
            negatives.append(sample)

    shuffle(positives), shuffle(negatives)
    return positives, negatives


def in_context_from_samples(samples, tiled=True, prompt_prefix="", prompt_sort_by=None):
    keys = ["input", "label"]
    positives = [samples[i] for i in range(len(samples)) if samples[i]["label"]]
    negatives = [samples[i] for i in range(len(samples)) if not samples[i]["label"]]
    prompt_order = []
    if not tiled:
        for p, n in zip(positives, negatives):
            prompt_order.append(p)
            prompt_order.append(n)
    else:
        prompt_order = positives + negatives

    prompt_samples = []
    for sample in prompt_order:
        prompt_samples.append({key: sample[key] for key in keys})

    prompt = prompt_prefix + pformat(prompt_samples)
    return prompt

def test_prompt_from_samples(positives, negatives, user_prefix="", tiled=False):
    samples = positives + negatives
    if not tiled:
        shuffle(samples)
    prompt_samples = [{"input": sample["input"]} for sample in samples]
    return user_prefix + pformat(prompt_samples), samples

json_prefix = "Please label the following inputs. Respond in JSON format like the examples given to you above."
def gpt_prediction(system_prompt, user_query, model="gpt-4", temperature=1.0, json_mode=False):
    if json_mode and model != "gpt-4-1106-preview":
        warn("json_mode only supported for gpt-4-1106-preview")
        warn("changing model to gpt-4-1106-preview")
        model = "gpt-4-1106-preview"
    if json_mode and "json" not in user_query.lower():
        warn("json_mode is on but user_query does not contain \"json\"")
        warn(f"adding prefix to user_query:\n\"{json_prefix}\"")
        user_query = json_prefix + user_query

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=temperature,
        max_tokens=4096, # max for the 1106-preview model
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object" if json_mode else "text"},
        seed=seed
    )
    # check if a response is an error
    # if response
    if response.choices[0].finish_reason == "length":
        warn("Response reached max tokens, consider increasing max_tokens")

    try:
        response_json = json.loads(response.choices[0].message.content)
        if type(response_json) != list:
            response_json = response_json[list(response_json.keys())[0]]
        if type(response_json) != list:
            warn("Response is not a list")
    except json.decoder.JSONDecodeError:
        warn("Response is not JSON")
        response_json = response.choices[0].message.content

    return response_json

def eval_response(response_json, test_samples):
    eval_results = []
    correct, mismatch, incorrect, corrupted = 0, 0, 0, 0
    for r_sample, t_sample in zip(response_json, test_samples):
        if type(r_sample) != dict or "label" not in r_sample or "input" not in r_sample:
            warn("Response sample is not a dict or does not contain \"label\" and \"input\"")
            warn(f"Response sample: {r_sample}")
            corrupted += 1
            continue
        r_sample["label"] = bool(r_sample["label"])
        eval_results.append(t_sample)
        eval_results[-1]["mismatch"] = r_sample["input"] != t_sample["input"]
        eval_results[-1]["eval"] = r_sample["label"] == t_sample["label"]
        eval_results[-1]["eval"] &= not eval_results[-1]["mismatch"]
        correct += eval_results[-1]["eval"]
        mismatch += eval_results[-1]["mismatch"]
        incorrect += not eval_results[-1]["eval"] and not eval_results[-1]["mismatch"]
    if correct + mismatch + incorrect + corrupted != len(test_samples):
        warn("Something went wrong with eval_response. \
             The sum of correct, mismatch, and incorrect should equal the number of test samples.")

    conf_matrix = [[0, 0], [0, 0]]
    for sample in eval_results:
        if sample["mismatch"]:
            continue
        conf_matrix[int(sample["label"])][int(sample["eval"])] += 1

    summary_dict = {
        "correct": correct,
        "mismatch": mismatch,
        "incorrect": incorrect,
        "corrupted": corrupted,
        "total": len(test_samples),
        "accuracy": correct / len(test_samples),
        "precision": conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1]),
        "recall": conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]),
    }
    return eval_results, summary_dict

def write_test_data(positives, negatives, samples_per_label, output_dir, unbalanced=False, max_test_ct=None, tiled=False, prompt_prefix="", prompt_sort_by=None):
    in_context_samples = positives[:samples_per_label] + negatives[:samples_per_label]
    in_context_prompt = in_context_from_samples(in_context_samples, tiled=tiled, prompt_prefix=prompt_prefix,
                                                prompt_sort_by=prompt_sort_by)
    with open(output_dir / f"in_context_prompt_{samples_per_label}.txt", "w") as f:
        f.write(in_context_prompt)

    if unbalanced:
        if max_test_ct is None:
            num_test_pos = len(positives) - samples_per_label
            num_test_neg = len(negatives) - samples_per_label
        else:
            num_test_pos = min(len(positives) - samples_per_label, max_test_ct)
            num_test_neg = min(len(negatives) - samples_per_label, max_test_ct)
    else:
        num_test_samples = min(len(positives), len(negatives)) - samples_per_label
        if not max_test_ct is None:
            num_test_samples = min(num_test_samples, max_test_ct)
        num_test_pos, num_test_neg = num_test_samples, num_test_samples
    test_positives = positives[samples_per_label:][:num_test_pos]
    test_negatives = negatives[samples_per_label:][:num_test_neg]
    test_prompt, test_samples = test_prompt_from_samples(test_positives, test_negatives)
    with open(output_dir / f"test_prompt_{samples_per_label}.txt", "w") as f:
        f.write(test_prompt)
    json.dump(test_samples, open(output_dir / f"test_samples_{samples_per_label}.json", "w"), indent=2)