# modify from DepictQA/build_datasets/scripts/add_distortion_refA_sd.py
import os
import glob
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageFile
import sys
sys.path.append("/home/dzc/yuanhao/syn_aes_data/utils")
sys.path.append("/home/dzc/yuanhao/syn_aes_data/DepictQA")

from constant import DIST_DICT, CATEGORY_TO_CLASSES
from tool import (
    get_category_from_class,
    get_distortion_class,
    save_json_append,
    seed_everything,
    weighted_sample_without_replacement,
    CATEGORY_WEIGHTS
)
from build_datasets.x_distortion import add_distortion

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description="Adding Distortion to the Reference Image")
parser.add_argument(
    "--reference_dir",
    type=str,
    required=True,
    help="Path to the reference image",
)
parser.add_argument(
    "--distortion_dir",
    type=str,
    required=True,
    help="Path to the distortion image",
)
parser.add_argument(
    "--seed",
    type=int,
    default=131,
    help="Random seed for reproducibility",
)
parser.add_argument(
    "-N",
    "--num_samples",
    type=int,
    default=5,
    help="Number of samples to generate per ref image",
)
parser.add_argument(
    "--json_path",
    type=str,
    required=True,
    help="Path to save the meta json file",
)



def select_samples(num_samples, excluded_categories=None):
    # if None, set an empty set
    excluded_categories = excluded_categories or set()

    category_pool = {}
    for category, classes in CATEGORY_TO_CLASSES.items():
        if category in excluded_categories:
            continue
        valid_classes = [cls for cls in classes if DIST_DICT.get(cls)]
        if valid_classes:
            category_pool[category] = valid_classes

    if not category_pool:
        for category, classes in CATEGORY_TO_CLASSES.items():
            valid_classes = [cls for cls in classes if DIST_DICT.get(cls)]
            if valid_classes:
                category_pool[category] = valid_classes

    if not category_pool:
        return []

    categories = list(category_pool.keys())
    weight_map = {cat: CATEGORY_WEIGHTS.get(cat, 1) for cat in categories}

    if num_samples <= len(categories):
        selected_categories = weighted_sample_without_replacement(categories, weight_map, num_samples)
    else:
        selected_categories = weighted_sample_without_replacement(categories, weight_map, len(categories))
        remaining = num_samples - len(selected_categories)
        if categories:
            weights = [weight_map.get(cat, 1) for cat in categories]
            if sum(weights) <= 0:
                selected_categories.extend(random.choices(categories, k=remaining))
            else:
                selected_categories.extend(random.choices(categories, weights=weights, k=remaining))

    sampled_funcs = []
    for category in selected_categories:
        classes = category_pool.get(category)
        if not classes:
            continue
        distortion_class = random.choice(classes)
        funcs = DIST_DICT.get(distortion_class)
        if funcs:
            sampled_funcs.append(random.choice(funcs))

    return sampled_funcs

if __name__ == "__main__":
    args = parser.parse_args()
    num_severity = 5
    resize = 768
    seed_everything(seed=args.seed)
    distortion_dir = args.distortion_dir
    os.makedirs(distortion_dir, exist_ok=True)
    # info about saving json
    json_file_path = os.path.join(args.json_path, "meta.json")
    processed_count = 0
    # collect image paths
    reference_dir = args.reference_dir
    img_types = ["*.png", "*.jpg", "*.jpeg","*.webp"]
    img_paths = []
    for img_type in img_types:
        img_paths.extend(sorted(glob.glob(os.path.join(reference_dir, img_type))))
    
    for idx_ref, img_path in enumerate(img_paths):
        print("=" * 100)
        print(f"Processing image {idx_ref + 1}/{len(img_paths)}: {os.path.basename(img_path)}")

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(img_path)[1] or ".png"
        img_lq_exist = os.path.exists(os.path.join(distortion_dir, f"{img_name}_0{img_ext}"))
        if img_lq_exist:
            print(f"{img_path} has been generated, skip.")
            continue

        # collect used distortion categories
        used_categories = set()
        # collect meta info about distortion
        dis_info_list = []
        
        for img_idx in range(args.num_samples):
            idx = 0
            while True:
                save_path = os.path.join(distortion_dir, f"{img_name}_{idx}{img_ext}")
                if not os.path.exists(save_path):
                    break
                idx += 1

            distortion_func_names = select_samples(1, excluded_categories=used_categories)
            if not distortion_func_names:
                continue

            distortion_name = distortion_func_names[0]
            distortion_class = get_distortion_class(distortion_name)
            category = get_category_from_class(distortion_class)
            if category:
                used_categories.add(category)
            severity = random.randint(1, num_severity)

            img = Image.open(img_path).convert("RGB")
            h, w = img.height, img.width
            if resize < min(h, w):
                ratio = resize / min(h, w)
                h_new, w_new = round(h * ratio), round(w * ratio)
                img = img.resize((w_new, h_new), resample=Image.Resampling.BICUBIC)
            img = np.array(img)
            img_lq = add_distortion(img, severity=severity, distortion_name=distortion_name)
            img_lq = Image.fromarray(img_lq)
            img_lq.save(save_path)

            dis_info_list.append({
                "distortion_class": distortion_class,
                "distortion_name": distortion_name,
                "severity": severity,
                "img_lq": save_path
            })
            

        if dis_info_list:
            entry = {
                img_name: {
                    "image_path": img_path,
                    "distortion_num": len(dis_info_list),
                    "distortions": dis_info_list,
                }
            }
            save_json_append(json_file_path, entry)
            processed_count += 1
