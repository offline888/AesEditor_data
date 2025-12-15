import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusion3InstructPix2PixPipeline

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for UltraEdit")
    # Standardized Arguments
    parser.add_argument("--json-path", type=str, required=True, help="Path to jsonl file")
    parser.add_argument("--image-root", type=str, required=True, help="Root dir for input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Root dir for output images")
    parser.add_argument("--model-path", type=str, default="BleachNick/SD3_UltraEdit_w_mask", help="Model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1. Setup Model
    device="cuda"
    print(f"Loading UltraEdit (SD3) from {args.model_path}...")
    
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16
    ).to(device)

    # 2. Load Data
    data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples.")

    # 3. Inference Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for item in tqdm(data, desc="Processing"):
        rel_path = item.get('raw') or item.get('image_path')
        instruction = item.get('instruction') or item.get('prompt')
        
        target_rel = item.get('target') or rel_path
        output_path = os.path.join(args.output_dir, target_rel)
        
        if os.path.exists(output_path): continue

        input_path = os.path.join(args.image_root, rel_path)
        if not os.path.exists(input_path): continue
            
        try:
            # Preprocess (Resize to 512x512)
            original_image = Image.open(input_path).convert("RGB")
            img_input = original_image.resize((512, 512))
            mask_img = Image.new("RGB", img_input.size, (255, 255, 255)) # Blank mask

            # Inference
            generator = torch.Generator(device).manual_seed(args.seed)
            result = pipe(
                prompt=instruction,
                image=img_input,
                mask_img=mask_img,
                negative_prompt="",
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            # Post-process (Resize back)
            if result.size != original_image.size:
                result = result.resize(original_image.size, Image.Resampling.LANCZOS)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.save(output_path)
            
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

if __name__ == "__main__":
    main()