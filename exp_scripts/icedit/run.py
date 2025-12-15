import os
import json
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from diffusers import FluxFillPipeline

def main():
    parser = argparse.ArgumentParser(description="ICEdit Batch Inference")
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default='black-forest-labs/FLUX.1-Fill-dev')
    parser.add_argument("--lora-path", default='RiverZ/normal-lora')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Setup Model
    device="cuda"
    print(f"Loading ICEdit from {args.model_path}...")
    
    pipe = FluxFillPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if args.lora_path: 
        print(f"Loading LoRA: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
    pipe.to(device)

    # 2. Load Data
    data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples.")

    # 3. Inference Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    
    for item in tqdm(data, desc="Processing"):
        rel_path=item.get('raw') or item.get('image_path')
        instruction=item.get('instruction') or item.get('prompt')
        
        target_rel=item.get('target') or rel_path
        output_path=os.path.join(args.output_dir, target_rel)
        
        if os.path.exists(output_path): continue
        
        input_path = os.path.join(args.image_root, rel_path)
        if not os.path.exists(input_path): continue

        try:
            image = Image.open(input_path).convert("RGB")
            
            if image.size[0] != 512:
                scale = 512 / image.size[0]
                new_h = int((int(image.size[1] * scale) // 8) * 8)
                image = image.resize((512, new_h))
            
            w, h = image.size
            
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(image, (0, 0))
            combined.paste(image, (w, 0))
            
            mask_arr = np.zeros((h, w * 2), dtype=np.uint8)
            mask_arr[:, w:] = 255
            mask = Image.fromarray(mask_arr)
            
            full_prompt = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'
        
            generator = torch.Generator("cuda").manual_seed(args.seed)
            result = pipe(
                prompt=full_prompt,
                image=combined,
                mask_image=mask,
                height=h,
                width=w * 2,
                guidance_scale=50,
                num_inference_steps=28,
                generator=generator
            ).images[0]
            
            result = result.crop((w, 0, w * 2, h))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.save(output_path)

        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

if __name__ == "__main__":
    main()