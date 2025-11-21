import os
import re
import argparse
import json
import time
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


parser = argparse.ArgumentParser(description="Generate instructions for multiple distortions")
parser.add_argument("--meta_json", type=str, required=True, 
                    help="Path to load meta json file")
parser.add_argument("--save_json", type=str, required=True,
                    help="Path to save generated instructions jsonl file")
parser.add_argument("--model-path",type=str,help="Path to load qwen model",
                    default='Qwen/Qwen3-VL-8B-Instruct')

distortion_name_dict={
        'brightness_brighten_shift_HSV':'enhance the brightness of the image by shifting the V channel in HSV.',
        'brightness_brighten_shift_RGB':'enhance the brightness of the image by shifting the RGB values',
        'brightness_brighten_gamma_HSV':'enhance the brightness of the image by adjusting the V channel in HSV with a gamma function.',
        'brightness_brighten_gamma_RGB':'enhance the brightness of the image by adjusting the RGB values with a gamma function',
        'brightness_darken_shift_HSV':'reduce the brightness of the image by shifting the V channel in HSV.',
        'brightness_darken_shift_RGB':'reduce the brightness of the image by shifting the RGB values.',
        'brightness_darken_gamma_HSV':'reduce the brightness of the image by adjusting the V channel in HSV with a gamma function.',
        'brightness_darken_gamma_RGB':'reduce the brightness of the image by adjusting the RGB values with a gamma function.',
        'contrast_strengthen_scale':'enhance the contrast of the image by scaling.',
        'contrast_strengthen_stretch':'enhance the contrast of the image by stretching.',
        'contrast_weaken_scale':'reduce the contrast of the image by scaling.',
        'contrast_weaken_stretch':'reduce the contrast of the image by stretching.',
        'saturate_strengthen_HSV':'enhance the saturation of the image by scaling S channel in HSV.',
        'saturate_strengthen_YCrCb':'enhance the saturation of the image by scaling CrCb channel in YCrCb.',
        'saturate_weaken_HSV':'reduce the saturation of the image by scaling S channel in HSV.',
        'saturate_weaken_YCrCb':'reduce the saturation of the image by scaling CrCb channel in YCrCb.',
        'temperature_warm_RGB':'warm the image by increasing the red channel and decreasing the blue channel.',
        'temperature_cool_RGB':'cool the image by decreasing the red channel and increasing the blue channel.',
        'temperature_warm_LAB':'warm the image by increasing the B channel in LAB.',
        'temperature_cool_LAB':'cool the image by decreasing the B channel in LAB.',
        'tint_green_RGB':'adjust tint of the image by increasing the green channel and decreasing the red channel.',
        'tint_magenta_RGB':'adjust tint of the image by increasing the red channel and decreasing the green channel.',
        'tint_green_LAB':'adjust tint of the image by decreasing the A channel in LAB.',
        'tint_magenta_LAB':'adjust tint of the image by increasing the A channel in LAB.',
        'exposure_increase_LAB':'increase the exposure of the image by increasing the L channel in LAB.',
        'exposure_decrease_LAB':'decrease the exposure of the image by decreasing the L channel in LAB.',
        'oversharpen':'over-sharpen the image by applying unsharp masking.',
        'decrease_sharpness':'decrease sharpness of the image.'
    }
opposite_distortion_name_dict = {
        'brightness_brighten_shift_HSV':'brightness_darken_shift_HSV',
        'brightness_brighten_shift_RGB':'brightness_darken_shift_RGB',
        'brightness_brighten_gamma_HSV':'brightness_darken_gamma_HSV',
        'brightness_brighten_gamma_RGB':'brightness_darken_gamma_RGB',
        'brightness_darken_shift_HSV':'brightness_brighten_shift_HSV',
        'brightness_darken_shift_RGB':'brightness_brighten_shift_RGB',
        'brightness_darken_gamma_HSV':'brightness_brighten_gamma_HSV',
        'brightness_darken_gamma_RGB':'brightness_brighten_gamma_RGB',
        'contrast_strengthen_scale':'contrast_weaken_scale',
        'contrast_strengthen_stretch':'contrast_weaken_stretch',
        'contrast_weaken_scale':'contrast_strengthen_scale',
        'contrast_weaken_stretch':'contrast_strengthen_stretch',
        'saturate_strengthen_HSV':'saturate_weaken_HSV',
        'saturate_strengthen_YCrCb':'saturate_weaken_YCrCb',
        'saturate_weaken_HSV':'saturate_strengthen_HSV',
        'saturate_weaken_YCrCb':'saturate_strengthen_YCrCb',
        'temperature_warm_RGB':'temperature_cool_RGB',
        'temperature_cool_RGB':'temperature_warm_RGB',
        'temperature_warm_LAB':'temperature_cool_LAB',
        'temperature_cool_LAB':'temperature_warm_LAB',
        'tint_green_RGB':'tint_magenta_RGB',
        'tint_magenta_RGB':'tint_green_RGB',
        'tint_green_LAB':'tint_magenta_LAB',
        'tint_magenta_LAB':'tint_green_LAB',
        'exposure_increase_LAB':'exposure_decrease_LAB',
        'exposure_decrease_LAB':'exposure_increase_LAB',
        'oversharpen':'decrease_sharpness'
    }

def generate_instruction(model, processor, img_ref_path, img_lq_path, distortion_name, severity):
    grades = ["slight distortion", "moderate distortion", "noticeable distortion", "severe distortion", "extreme distortion"]
    bea_grades = ["mild enhancement", "moderate enhancement", "notable enhancement", "significant enhancement", "dramatic enhancement"]

    sev_int = max(1, min(5, int(severity)))
    
    dist_info_text = str(
        "The first image is the distorted image, and the second image is the clean version. "
        "The first image is produced by adding distortion to the second image. "
        "The distortion information and beautification operation information is as follows:\n"
        f"- Distortion type: {distortion_name_dict[distortion_name]}\n"
        f"- Severity level: {grades[sev_int-1]} (severity {sev_int} out of 5)\n"
    )
    reverse_key = opposite_distortion_name_dict.get(distortion_name)
    if reverse_key:
        dist_info_text += f"- Beautification operation description: {distortion_name_dict[reverse_key]}\n"
        dist_info_text += f"- Beautification operation severity: {bea_grades[sev_int-1]} (severity {sev_int} out of 5)\n"
    
    query_text = str(
        "Given a distorted image, its clean version, and the corresponding distortion information (type and severity), analyze the visual changes. "
        "Infer how the user wants to clean up the distorted image and express it concisely.\n"
        "Please provide FOUR possible instructions of the user to beautify the image. Instruction 1 and 2 are non‑professional instructions; Instruction 3 and 4 are professional instructions.\n"
        "Special Notes For Non‑professional Instructions (1 and 2):\n"
        "1. Simulate the voice of a real non‑professional user; use casual, non‑technical terms. As if you were the decision maker, explain operations to a trusted partner, not just fill a template.\n"
        "2. Based on the described beautification operation, generate beautification instructions in the opposite direction of the distortion.\n"
        "3. Translate the beautification operation into relatable visual outcomes without numbers or technical terms.\n"
        "4. Base the instructions on global adjustments rather than specific local areas.\n"
        "5. Keep each within 40 words.DON'T use useless interjections.\n\n"
        "Special Notes For Professional Instructions (3 and 4):\n"
        "1. Simulate the voice of an expert in image beautification, stating edits rigorously. As if you were the decision maker, explain operations to a trusted partner, not just fill a template.\n"
        "2. Based on the described beautification operation, generate beautification instructions in the opposite direction of the distortion.\n"
        "3. Express the beautification operation with professional terms and appropriate severity,"
        "but DON'T use technical terms like 'operate on xxx channel in xxx color space',' xxx level in xxx color space'...etc."
        "You shouldn't describe with underlying technical details when explaining.\n"
        "4. Base the instructions on global adjustments rather than specific local areas.\n"
        "5. Keep each within 40 words.DON'T use useless interjections.\n"
        "Output requirements:\n"
        "- Provide FOUR instructions to beautify the image.\n"
        "- Use varied sentence patterns to keep instructions concise; do not repeat one pattern.\n"
        "- The following output format MUST be strictly followed:\n"
        '{\n'
        '  "instruction 1": "< The first instruction of the non-professional user >",\n'
        '  "instruction 2": "< The second instruction of the non-professional user >",\n'
        '  "instruction 3": "< The first instruction of the professional user >",\n'
        '  "instruction 4": "< The second instruction of the professional user >"\n'
        '}\n\n'
    )
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": img_lq_path},
            {"type": "image", "url": img_ref_path},
            {'type': "text" , "text": dist_info_text+query_text},
        ]
    }]  
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0] if output_text else ""
    

    json_match = re.search(r'\{[^{}]*"instruction 1"[^{}]*"instruction 2"[^{}]*"instruction 3"[^{}]*"instruction 4"[^{}]*\}',response,re.DOTALL)
    if json_match:
        matched_text = json_match.group(0)
        try:
            return json.loads(matched_text)
        except json.JSONDecodeError as e:
            print(f"JSON decode error:{e}")
            # find valid instructions from response
            cleaned=matched_text.strip()
            cleaned=re.sub(r',\s*}', '}', cleaned)  
            cleaned=re.sub(r',\s*]', ']', cleaned)
            cleaned=re.sub(r'"instruction 1":\s*"([^"]*)"\s*\n\s*"instruction 2"', r'"instruction 1": "\1",\n  "instruction 2"', cleaned)
            cleaned=re.sub(r'"instruction 3":\s*"([^"]*)"\s*\n\s*"instruction 4"', r'"instruction 3": "\1",\n  "instruction 4"', cleaned)
            return json.loads(cleaned)
    else:
        return {
            "instruction 1": "",
            "instruction 2": "",
            "instruction 3": "",
            "instruction 4": ""
        }


def load_existing_results(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            return set(json.loads(line)['pair_id'] for line in f)
    return set()


def save_results(save_path, results):
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    args = parser.parse_args()
    # load model
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    with open(args.meta_json, 'r', encoding='utf-8') as f:
        distortion_data = json.load(f)
    
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    main_results_path = args.save_json
    all_results = load_existing_results(main_results_path)
    
    total_entries=sum(
        len(image_data.get("distortions", []))
        for image_data in distortion_data.values()
    )
    
    skipped_count = 0
    failed_count = 0
    processed_count = 0

    print(f"=" * 80)
    print(f"Total entries: {total_entries}")
    print(f"Already processed: {len(all_results)}")
    print(f"Remaining: {total_entries - len(all_results)}")
    print(f"=" * 80)
    
    for image_name, image_data in distortion_data.items():
        img_ref = image_data.get("image_path")
        if not img_ref:
            continue
            
        distortions = image_data.get("distortions", [])
        for dist_info in distortions:
            try:
                img_lq = dist_info.get('img_lq', '')
                if not img_lq:
                    continue
                    
                entry_key = f"{image_name}_{os.path.splitext(os.path.basename(img_lq))[0]}"
                
                if entry_key in all_results:
                    skipped_count += 1
                    continue
                
                # 记录开始时间
                start_time = time.time()
                
                instructions=generate_instruction(
                        model, processor,
                        img_ref,
                        dist_info.get("img_lq"),
                        dist_info.get("distortion_name"),
                        dist_info.get("severity")
                )
                if not isinstance(instructions, dict):
                    instructions = {}
                if instructions.get("instruction 1", "") == "" or instructions.get("instruction 2", "") == "" or instructions.get("instruction 3", "") == "" or instructions.get("instruction 4", "") == "":
                    failed_count += 1
                    print(f"Failed to generate instructions for {entry_key}")
                    continue
                    
                output_entry = {
                        "pair_id": entry_key,
                        "image_name": image_name,
                        "img_ref": img_ref,
                        "img_lq": dist_info.get("img_lq"),
                        "distortion_class": dist_info.get("distortion_class"),
                        "distortion_name": dist_info.get("distortion_name"),
                        "severity": dist_info.get("severity"),
                        "instruction_1": instructions.get("instruction 1", ""),
                        "instruction_2": instructions.get("instruction 2", ""),
                        "instruction_3": instructions.get("instruction 3", ""),
                        "instruction_4": instructions.get("instruction 4", ""),
                }

                save_results(main_results_path, output_entry)
            except Exception as e:
                failed_count += 1
                print(f"Failed to generate instructions for {entry_key}: {e}")
                continue
            
            processed_count += 1
            # 计算并输出耗时
            elapsed_time = time.time() - start_time
            print(f"Processed {processed_count + skipped_count + failed_count}/{total_entries}: {entry_key} (Time: {elapsed_time:.2f}s)")
           
    
    print(f"=" * 80)
    print(f"Processing completed!")
    print(f"Total: {total_entries}")
    print(f"Processed in this run: {processed_count}")
    print(f"Already processed: {skipped_count + failed_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to: {main_results_path}")
    print(f"=" * 80)