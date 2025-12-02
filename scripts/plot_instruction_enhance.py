import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{json_path} is empty or not a dict.")
    return data


def select_samples(
    data: Dict[str, Any], count: int, seed: Optional[int] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    keys = list(data.keys())
    if not keys:
        raise ValueError("instructions.json has no entries.")

    count = max(1, min(count, len(keys)))
    rng = random.Random(seed)
    sampled_keys = rng.sample(keys, k=count)
    return [(key, data[key]) for key in sampled_keys]


def resolve_image_path(path_str: str, root_dir: str) -> str:
    if not path_str:
        raise ValueError("Image path is empty.")
    if os.path.isabs(path_str) or not root_dir:
        return path_str
    return os.path.join(root_dir, path_str)


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def split_numbered_instructions(text: str) -> List[str]:
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)[\.\)]\s*(.*)$", line)
        if match:
            lines.append(match.group(2).strip())
        else:
            lines.append(line)
    return lines


def draw_instruction_panel(ax, key: str, item: Dict[str, Any]) -> None:
    ax.axis("off")

    title = key

    ax.text(
        0.5,
        0.94,
        title,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes,
    )

    instruction_dict = item.get("instruction", {}) or {}

    lines: List[str] = [
        "Non-professional:",
        f"1. {instruction_dict.get('instruction_1', '')}",
        f"2. {instruction_dict.get('instruction_2', '')}",
        "",
        "Professional:",
        f"3. {instruction_dict.get('instruction_3', '')}",
        f"4. {instruction_dict.get('instruction_4', '')}",
    ]

    ax.text(
        0.03,
        0.78,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=12,
        transform=ax.transAxes,
        wrap=True,
    )


def plot_pair(
    key: str,
    item: Dict[str, Any],
    image_root: str,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    raw_path = resolve_image_path(item.get("raw", ""), image_root)
    target_path = resolve_image_path(item.get("target", ""), image_root)

    img_raw = load_image(raw_path)
    img_target = load_image(target_path)

    fig = plt.figure(figsize=(14, 8), dpi=150, layout="constrained")
    grid = gridspec.GridSpec(2, 2, height_ratios=[1.2, 3], width_ratios=[1, 1])

    ax_top = fig.add_subplot(grid[0, :])
    draw_instruction_panel(ax_top, key, item)

    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])
    ax_left.imshow(img_raw)
    ax_right.imshow(img_target)
    ax_left.set_title("Raw", fontsize=16, fontweight="bold", pad=6)
    ax_right.set_title("Target", fontsize=16, fontweight="bold", pad=6)

    for ax in (ax_left, ax_right):
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#333333")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved to {output_path}")
    elif show:
        plt.show()

    plt.close(fig)


def sanitize_filename(name: str) -> str:
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "sample"


def main():
    parser = argparse.ArgumentParser(
        description="Sample entries from real_results instructions and visualize raw vs. target with instructions."
    )
    parser.add_argument(
        "--data_json",
        type=str,
        required=True,
        help="Path to real_results/instructions.json",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Root directory to prepend to relative image paths",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=32,
        help="Number of samples to display",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./plots_real",
        help="Directory to save the plots (ignored if --show is used)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of saving",
    )
    args = parser.parse_args()

    data = load_json(args.data_json)
    samples = select_samples(data, args.N, args.seed)

    if args.show:
        for key, item in samples:
            print(f"[show] plotting {key}")
            plot_pair(key, item, args.image_root, output_path=None, show=True)
    else:
        save_dir = Path(args.save_dir)
        for idx, (key, item) in enumerate(samples, start=1):
            filename = f"{idx:04d}_{sanitize_filename(key)}.png"
            print(f"[save] ({idx}/{len(samples)}) plotting {key}")
            plot_pair(key, item, args.image_root, save_dir / filename, show=False)


if __name__ == "__main__":
    main()


