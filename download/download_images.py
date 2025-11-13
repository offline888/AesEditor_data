#!/usr/bin/env python3
import sys
from pathlib import Path

import requests


def download(url, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1] or "downloaded_file"
    filepath = target_dir / filename
    if filepath.exists():
        return
    response = requests.get(url, timeout=30, stream=True)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    script_dir = Path(__file__).parent
    urls_path = script_dir / "urls.txt"
    output_dir = script_dir.parent / "imgs"

    try:
        urls = [line.strip() for line in urls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        print(f"找不到 {urls_path}")
        sys.exit(1)

    for index, url in enumerate(urls, 1):
        try:
            download(url, output_dir)
            print(f"[{index}/{len(urls)}] 下载完成: {url}")
        except Exception as exc:
            print(f"[{index}/{len(urls)}] 下载失败: {url} -> {exc}")


if __name__ == "__main__":
    main()