#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
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
    parser = argparse.ArgumentParser(description="Download images from URLs")
    parser.add_argument("--urls_path", type=str, default="download/urls.txt", help="Path to the URLs file")
    parser.add_argument("--output_dir", type=str, default="data/HQ", help="Path to the output directory")
    parser.add_argument("--part", type=str, default="4-0", help="Part of the URLs file to download, e.g. 4-0 is the 0th part of the 4 parts")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    urls_path = Path(args.urls_path)
    output_dir = Path(args.output_dir)

    try:
        urls = [line.strip() for line in urls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        print(f"找不到 {urls_path}")
        sys.exit(1)
    
    parts, index = args.part.split("-")
    parts = int(parts)
    index = int(index)
    urls = urls[index::parts]

    for index, url in enumerate(urls, 1):
        try:
            download(url, output_dir)
            print(f"[{index}/{len(urls)}] 下载完成: {url}")
        except Exception as exc:
            print(f"[{index}/{len(urls)}] 下载失败: {url} -> {exc}")


if __name__ == "__main__":
    main()