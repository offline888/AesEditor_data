import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"{json_path} is empty or not a dict.")
    return data


def select_samples(
    data: Dict[str, Any],
    count: int,
    seed: Optional[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    keys = list(data.keys())
    if not keys:
        raise ValueError("instructions.json has no entries.")

    count = max(1, min(count, len(keys)))
    if seed is None:
        sampled_keys = random.sample(keys, k=count)
    else:
        rng = random.Random(seed)
        sampled_keys = rng.sample(keys, k=count)

    return [(key, data[key]) for key in sampled_keys]


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image file not found: {path}")
    return Image.open(path).convert("RGB")


def draw_instructions(ax, item: Dict[str, Any]) -> None:
    ax.axis("off")

    title = item.get("image_name", "")
    ax.text(
        0.5,
        0.95,
        title,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )

    lines = [
        item.get("instruction_1", ""),
        item.get("instruction_2", ""),
        item.get("instruction_3", ""),
        item.get("instruction_4", ""),
    ]
    text = "\n\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))
    ax.text(
        0.02,
        0.80,
        text,
        ha="left",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
        wrap=True,
    )


def build_distortion_title(item: Dict[str, Any]) -> str:
    distortion_names = item.get("distortion_names")
    severities = item.get("severities")

    if isinstance(distortion_names, list) and distortion_names:
        parts: List[str] = []
        for idx, name in enumerate(distortion_names):
            severity = ""
            if isinstance(severities, list) and idx < len(severities):
                severity = str(severities[idx])
            parts.append(f"{name}:{severity}" if severity else str(name))
        return " + ".join(parts)

    distortion_name = item.get("distortion_name", item.get("distortion_class", ""))
    severity = item.get("severity", "")
    return f"{distortion_name}:{severity}" if severity else str(distortion_name)


def plot_pair(item: Dict[str, Any], output_path: Optional[Path] = None) -> None:
    img_ref = load_image(item["img_ref"])
    img_lq = load_image(item["img_lq"])
    distortion_title = build_distortion_title(item)

    fig = plt.figure(figsize=(14, 8), dpi=150, layout="constrained")
    grid = gridspec.GridSpec(2, 2, height_ratios=[1.2, 3], width_ratios=[1, 1])

    ax_top = fig.add_subplot(grid[0, :])
    draw_instructions(ax_top, item)

    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])
    ax_left.imshow(img_ref)
    ax_right.imshow(img_lq)
    ax_left.set_title("Original", fontsize=14, fontweight="bold", pad=6)
    ax_right.set_title(distortion_title, fontsize=14, fontweight="bold", pad=6)

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
    else:
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
        description="Randomly sample entries and plot ref vs. distorted images with instructions."
    )
    parser.add_argument(
        "--single_json",
        type=str,
        required=True,
        help="Path to single dataset instructions.json",
    )
    parser.add_argument(
        "--multi_json",
        type=str,
        required=True,
        help="Path to multi dataset instructions.json",
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
        default=100,
        help="Number of samples to display for each dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./plots",
        help="Directory to save the plots",
    )
    args = parser.parse_args()

    data_single = load_json(args.single_json)
    data_multi = load_json(args.multi_json)

    samples_single = select_samples(data_single, args.N, args.seed)
    multi_seed = args.seed + 1 if args.seed is not None else None
    samples_multi = select_samples(data_multi, args.N, multi_seed)

    save_dir = Path(args.save_dir)
    single_dir = save_dir / "single"
    multi_dir = save_dir / "multi"

    for idx, (key, item) in enumerate(samples_single, start=1):
        print(f"[single] ({idx}/{len(samples_single)}) plotting {key}")
        filename = f"{idx:04d}_{sanitize_filename(key)}.png"
        plot_pair(item, single_dir / filename)

    for idx, (key, item) in enumerate(samples_multi, start=1):
        print(f"[multi] ({idx}/{len(samples_multi)}) plotting {key}")
        filename = f"{idx:04d}_{sanitize_filename(key)}.png"
        plot_pair(item, multi_dir / filename)


if __name__ == "__main__":
    main()


