import json
import random
import sys
import os
import argparse
from PIL import Image, ImageFile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constant import CATEGORY_TO_CLASSES
from utils.tool import (
    seed_everything,
    get_category_from_class,
    get_distortion_name,
    is_distortion_classes_duplicate,
)
from depictqa.build_datasets.scripts.constants_md import multi_distortions_dict
from depictqa.build_datasets.x_distortion import add_distortion, distortions_dict

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
    "-N1",
    "--num_multi_distortions",
    type=int,
    default=2,
    help="Number of distortions to combine on each image .",
)
parser.add_argument(
    "-N2",
    "--num_distortion_images",
    type=int,
    default=5,
    help="Number of distortion images to generate per reference image .",
)
parser.add_argument(
    "--json_path",
    type=str,
    required=True,
    help="Path to save the meta json file",
)

def select_samples(num_distortions, excluded_categories=None):
    all_classes = []
    for classes_in_category in CATEGORY_TO_CLASSES.values():
        for distortion_class in classes_in_category:
            all_classes.append(distortion_class)
    available_classes = list(dict.fromkeys(all_classes))

    if num_distortions == 1:
        print("Only support num_distortions > 1.")
        return

    if excluded_categories:
        candidate_classes = [
            cls
            for category, classes in CATEGORY_TO_CLASSES.items()
            if category not in excluded_categories
            for cls in classes
            if cls in available_classes
        ]
        first_class = random.choice(candidate_classes) if candidate_classes else random.choice(available_classes)
    else:
        first_class = random.choice(available_classes)

    selected_classes = [first_class]
    for _ in range(num_distortions - 1):
        last_class = selected_classes[-1]
        if last_class in multi_distortions_dict:
            compatible_classes = [
                c
                for c in multi_distortions_dict[last_class]
                if c in distortions_dict and c in available_classes and c not in selected_classes
            ]
            if compatible_classes:
                next_class = random.choice(compatible_classes)
            else:
                remaining = [c for c in available_classes if c not in selected_classes]
                next_class = random.choice(remaining) if remaining else random.choice(available_classes)
        else:
            remaining = [c for c in available_classes if c not in selected_classes]
            next_class = random.choice(remaining) if remaining else random.choice(available_classes)
        selected_classes.append(next_class)
    return selected_classes


if __name__ == "__main__":
    args = parser.parse_args()
    num_severity = 5
    resize = 768
    seed_everything(seed=args.seed)

    reference_dir = Path(args.reference_dir)
    img_types = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    img_paths = []
    for img_type in img_types:
        img_paths.extend(sorted(reference_dir.glob(img_type)))

    print(f"Total images to process: {len(img_paths)}")
    random.shuffle(img_paths)

    distortion_dir = Path(args.distortion_dir)
    distortion_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.json_path)
    if meta_path.suffix != ".json":
        meta_path = meta_path / "meta.json"
    summary_data = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            summary_data = json.load(f)
        print(f"Loaded existing summary with {len(summary_data)} entries")

    for idx_ref, img_path in enumerate(img_paths):
        print("=" * 100)
        print(f"Processing image {idx_ref + 1}/{len(img_paths)}: {img_path.name}")

        img_name = img_path.stem
        img_ext = img_path.suffix or ".png"
        existing_entry = summary_data.get(img_name)
        if existing_entry:
            expected = existing_entry.get("distortion_num", 0)
            distortions_complete = all(
                (distortion_dir / f"{img_name}_{i}{img_ext}").exists()
                for i in range(expected)
            )
            if distortions_complete:
                print(f"{img_path} has been generated, skip.")
                continue

        used_categories = set()
        distortions_list = []

        for img_idx in range(args.num_distortion_images):
            idx = 0
            while True:
                save_path = distortion_dir / f"{img_name}_{idx}{img_ext}"
                if not save_path.exists():
                    break
                idx += 1

            distortion_entry = None
            max_retries = 50
            for _ in range(max_retries):
                distortion_classes = select_samples(
                    args.num_multi_distortions, excluded_categories=used_categories
                )
                if distortion_classes is None:
                    continue
                if is_distortion_classes_duplicate(distortion_classes, distortions_list):
                    continue

                first_class = (
                    distortion_classes[0] if isinstance(distortion_classes, list) else distortion_classes
                )
                category = get_category_from_class(first_class)
                if category:
                    used_categories.add(category)

                img = Image.open(img_path).convert("RGB")
                h, w = img.height, img.width
                if resize < min(h, w):
                    ratio = resize / min(h, w)
                    h_new, w_new = round(h * ratio), round(w * ratio)
                    img = img.resize((w_new, h_new), resample=Image.Resampling.BICUBIC)
                img_lq = np.array(img)

                severities = []
                distortion_order_name = {}
                distortion_classes_list = (
                    distortion_classes if isinstance(distortion_classes, list) else [distortion_classes]
                )
                for order_idx, distortion_class in enumerate(distortion_classes_list):
                    sampled_distortion = get_distortion_name(distortion_class)
                    severity = random.randint(1, num_severity)
                    severities.append(severity)
                    distortion_order_name[sampled_distortion] = order_idx
                    img_lq = add_distortion(
                        img_lq, severity=severity, distortion_name=sampled_distortion
                    )

                Image.fromarray(img_lq).save(save_path)
                distortion_entry = {
                    "distortion_classes": distortion_classes_list,
                    "distortion_order_name": distortion_order_name,
                    "severities": severities,
                    "img_lq": str(save_path),
                }
                break

            if distortion_entry is None:
                print(
                    f"Warning: After {max_retries} retries, failed to generate unique distortion for {img_name}_{idx}."
                )
                continue

            distortions_list.append(distortion_entry)

        if distortions_list:
            summary_data[img_name] = {
                "img_path": str(img_path),
                "distortion_num": len(distortions_list),
                "distortions": distortions_list,
            }

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(summary_data, f, indent=4)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"\nFinal summary saved with {len(summary_data)} entries")
