import json
from warnings import warn

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

default_methods = ["positive", "negative", None]
def create_dataset(dataset, tags_true, tags_false=[], tags_common=[], default_method="positive"):
    tags_true, tags_false, tags_common = set(tags_true), set(tags_false), set(tags_common)
    if len(tags_true & tags_false) > 0:
        warn(f"Tags in both tags_true and tags_false: {tags_true & tags_false}")

    tags_true = tags_true | tags_common
    tags_false = tags_false | tags_common

    positives, negatives = [], []
    for sample in dataset:
        positive = all([tag in sample["tags"] for tag in tags_true])
        negative = all([tag in sample["tags"] for tag in tags_false])
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
            positives.append(sample)
        elif negative:
            negatives.append(sample)

    return positives, negatives