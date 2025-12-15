import os
import sys
import json
import torch
import argparse
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline

# Ensure FlowEdit_utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flowedit_utils import FlowEditFLUX

def main():
    parser = argparse.ArgumentParser(description="Inference for FlowEdit")
    parser.add_argument("--json-path", type=str, required=True, help="Path to jsonl file")
    parser.add_argument("--image-root", type=str, required=True, help="Root dir for input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Root dir for output images")
    parser.add_argument("--model-path", type=str, default="black-forest-labs/FLUX.1-dev", help="Model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1. Setup pipeline
    device="cuda"
    print(f"Loading FLUX from {args.model_path}")
    
    pipe=FluxPipeline.from_pretrained(args.model_path, 
                                      torch_dtype=torch.bfloat16)
    pipe.to(device)
    scheduler=pipe.scheduler
    # infer config
    t_steps = 28
    src_guidance = 1.5
    tar_guidance = 5.5
    n_avg = 1
    n_min = 0
    n_max = 24

    # 2. Load Data
    data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")

    # 3. Inference Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Global Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for item in tqdm(data, desc="Processing"):
        rel_path = item.get('raw') or item.get('image_path')
        src_prompt = item.get('source_prompt', '') 
        tar_prompt = item.get('instruction') or item.get('prompt')
        
        target_rel = item.get('target') or rel_path
        output_path = os.path.join(args.output_dir, target_rel)
        
        if os.path.exists(output_path):continue
        
        input_path = os.path.join(args.image_root, rel_path)
        if not os.path.exists(input_path):continue

        try:
            image = Image.open(input_path).convert("RGB")
            orig_size = image.size
            w, h = image.size
            image = image.crop((0, 0, w - w % 16, h - h % 16)) # Crop for VAE
            
            image_src = pipe.image_processor.preprocess(image)
            image_src = image_src.to(device, dtype=pipe.dtype)

            # Encode
            with torch.autocast("cuda", dtype=pipe.dtype), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to(device)

            # Inference
            x0_tar = FlowEditFLUX(
                pipe, scheduler, x0_src,
                src_prompt, tar_prompt, "",
                T_steps=t_steps, n_avg=n_avg,
                src_guidance_scale=src_guidance, tar_guidance_scale=tar_guidance,
                n_min=n_min, n_max=n_max
            )

            # Decode
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda", dtype=pipe.dtype), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            
            image_tar = pipe.image_processor.postprocess(image_tar)[0]

            if image_tar.size != orig_size:
                image_tar = image_tar.resize(orig_size, Image.Resampling.LANCZOS)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image_tar.save(output_path)

        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

if __name__ == "__main__":
    main()