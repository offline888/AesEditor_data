from constant import CATEGORY_TO_CLASSES, DIST_DICT
import os
import json
import random
import numpy as np

def get_category_from_class(distortion_class):
    for category, classes in CATEGORY_TO_CLASSES.items():
        if distortion_class in classes:
            return category
    return None

def get_distortion_name(distortion_name):
    all_names = [name for names in DIST_DICT.values() for name in names]
    if distortion_name in all_names:
        return distortion_name
    if distortion_name in DIST_DICT:
        return random.choice(DIST_DICT[distortion_name])
    key = random.choice(list(DIST_DICT.keys()))
    return random.choice(DIST_DICT[key])

def get_distortion_class(distortion_name):
    for key in DIST_DICT:
        if distortion_name in DIST_DICT[key]:
            return key

def seed_everything(seed=131):
    np.random.seed(seed)
    random.seed(seed**2)

def save_json_append(json_file_path, new_data):
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as fr:
            existing_data = json.load(fr)
        existing_data.update(new_data)
        data_to_save = existing_data
    else:
        data_to_save = new_data
    
    os.makedirs(os.path.dirname(json_file_path) if os.path.dirname(json_file_path) else ".", exist_ok=True)
    with open(json_file_path, "w") as fw:
        json.dump(data_to_save, fw, indent=4)
    return data_to_save

def compute_category_weights():
    weights = {}
    for category, classes in CATEGORY_TO_CLASSES.items():
        weight = 0
        for cls in classes:
            funcs = DIST_DICT.get(cls, [])
            weight += len(funcs)
        weights[category] = weight if weight > 0 else 1
    return weights

CATEGORY_WEIGHTS = compute_category_weights()

def weighted_sample_without_replacement(categories, weight_map, k):
    selected = []
    available = list(categories)
    for _ in range(min(k, len(available))):
        total_weight = sum(weight_map.get(cat, 0) for cat in available)
        if total_weight <= 0:
            # fall back to uniform random choice
            choice = random.choice(available)
        else:
            threshold = random.uniform(0, total_weight)
            cumulative = 0.0
            choice = available[0]
            for cat in available:
                cumulative += weight_map.get(cat, 0)
                if threshold <= cumulative:
                    choice = cat
                    break
        selected.append(choice)
        available.remove(choice)
    return selected
    
def distortion_classes_equal(classes1, classes2):
    list1 = classes1 if isinstance(classes1, list) else [classes1]
    list2 = classes2 if isinstance(classes2, list) else [classes2]
    return sorted(list1) == sorted(list2)


def is_distortion_classes_duplicate(new_classes, distortions_list):
    for item in distortions_list:
        existing_classes = item.get("distortion_classes", [])
        if distortion_classes_equal(new_classes, existing_classes):
            return True
    return False