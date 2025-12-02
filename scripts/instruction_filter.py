import os
import re
import argparse
import json
import time
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


parser = argparse.ArgumentParser(description="Generate beautification instructions from raw/target pairs")
parser.add_argument("--data_json", type=str, required=True,
                    help="Path to JSONL file containing raw/target pairs and instructions")
parser.add_argument("--save_json", type=str, required=True,
                    help="Path to save generated instructions jsonl file")
parser.add_argument("--model-path",type=str,help="Path to load qwen model",
                    default='Qwen/Qwen3-VL-8B-Instruct')
parser.add_argument("--image_root", type=str, required=True,
                    help="Optional root directory to prepend to relative raw/target paths")
parser.add_argument("--part", type=str, required=True, help="Part of the data to process, 8-0 means the 0th part of the 8 parts")


def build_instruction_block(instruction_text):
    cleaned = (instruction_text or "").strip()
    if not cleaned:
        return "Instruction candidates provided by the user: (missing)\n\n"

    candidates = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\d+)[\.\)]\s*(.*)$', line)
        text = match.group(2).strip() if match else line
        candidates.append(text)
        # if len(candidates) == 5:
        #     break

    structured = [
        {"id": idx + 1, "text": text}
        for idx, text in enumerate(candidates)
    ]
    if not structured:
        return "Instruction candidates provided by the user: (missing)\n\n"

    structured_json = json.dumps(structured, ensure_ascii=False, indent=2)
    return (
        "Instruction candidates provided by the user (JSON array of id/text):\n"
        f"{structured_json}\n\n"
    )


def filter_data(model, processor, img_raw_path, img_target_path, instruction_block):
    query_text = str(
        "Given an original image, its beautified version, and multiple corresponding beautification instructions listed above, analyze the visual differences between the two images.\n"
        "First, evaluate the instructions and select the two indices that most accurately describe the observed beautification process. Avoid overly generic guidance; focus on instructions that clearly match this transformation.\n"
        "Second, using ONLY the two selected instructions as inspiration, craft FOUR new beautification instructions from different perspectives: instructions 1 and 2 must sound non-professional, while instructions 3 and 4 must sound professional.\n"
        "Special Notes for Filtering:\n"
        "1. Always review all instructions before selecting the two that best capture the beautification process.\n"
        "2. Do not choose instructions that are vague or universally applicable; the selection must reflect the unique characteristics of this edit.\n\n"
        "Special Notes For Non-professional Instructions (1 and 2):\n"
        "1. Simulate the voice of a real non-professional user; use casual, relatable descriptions.\n"
        "2. Rephrase the selected beautification intent in simple, outcome-focused language without technical jargon.\n"
        "3. Focus on global tonal or structural guidance unless a local tweak is visually obvious.\n"
        "4. Keep each within 40 words. DON'T use filler interjections.\n\n"
        "Special Notes For Professional Instructions (3 and 4):\n"
        "1. Simulate the voice of an expert in image beautification, stating edits rigorously yet succinctly.\n"
        "2. Describe the improvements inferred from the selected instructions with professional terminology but without low-level color-space operations.\n"
        "3. Focus on global tonal or structural guidance unless a local tweak is visually obvious.\n"
        "4. Keep each within 40 words. DON'T use filler interjections.\n"
        "Output requirements:\n"
        "- Provide the two selected instruction indices followed by FOUR new beautification instructions.\n"
        "- Use varied sentence patterns to keep instructions concise; do not repeat one pattern.\n"
        "- The following output format MUST be strictly followed:\n"
        '{\n'
        '  "select_ins": [good_beauty_instruction_index1, good_beauty_instruction_index2],\n'
        '  "instruction 1": "< The first beautification instruction of the non-professional user >",\n'
        '  "instruction 2": "< The second beautification instruction of the non-professional user >",\n'
        '  "instruction 3": "< The first beautification instruction of the professional user >",\n'
        '  "instruction 4": "< The second beautification instruction of the professional user >"\n'
        '}\n\n'
    )
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": img_raw_path},
            {"type": "image", "url": img_target_path},
            {'type': "text", "text": instruction_block + query_text},
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
    

    json_match = re.search(
        r'\{[^{}]*"select_ins"[^{}]*"instruction 1"[^{}]*"instruction 2"[^{}]*"instruction 3"[^{}]*"instruction 4"[^{}]*\}',
        response,
        re.DOTALL
    )
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
            "select_ins": [],
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
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    with open(args.data_json, 'r', encoding='utf-8') as f:
        data_entries = [json.loads(line) for line in f if line.strip()]
    
    parts, index = args.part.split("-")
    parts = int(parts)
    index = int(index)
    data_entries = data_entries[index::parts]

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    main_results_path = args.save_json
    all_results = load_existing_results(main_results_path)

    total_entries = len(data_entries)
    skipped_count = 0
    failed_count = 0
    processed_count = 0

    print("=" * 80)
    print(f"Total entries: {total_entries}")
    print(f"Already processed: {len(all_results)}")
    print(f"Remaining: {total_entries - len(all_results)}")
    print("=" * 80)

    for entry in data_entries:
        entry_key = entry.get("pair_id")
        if not entry_key:
            continue
        if entry_key in all_results:
            skipped_count += 1
            continue

        try:
            raw_path = entry.get("raw", "") or ""
            target_path = entry.get("target", "") or ""
            if args.image_root:
                if raw_path and not os.path.isabs(raw_path):
                    raw_path = os.path.join(args.image_root, raw_path)
                if target_path and not os.path.isabs(target_path):
                    target_path = os.path.join(args.image_root, target_path)

            instruction_text = entry.get("instructions", "")
            instruction_block = build_instruction_block(instruction_text)
            
            start_time = time.time()

            instructions = filter_data(
                model=model,
                processor=processor,
                img_raw_path=raw_path,
                img_target_path=target_path,
                instruction_block=instruction_block,
            )
            if not isinstance(instructions, dict):
                instructions = {}
            
            if instructions.get("instruction 1", "") == "" or instructions.get("instruction 2", "") == "" or instructions.get("instruction 3", "") == "" or instructions.get("instruction 4", "") == "":
                 failed_count += 1
                 print(f"Failed to generate instructions for {entry_key}")
                 continue

            select_ins = instructions.get("select_ins", [])
            synthesized_instructions = [
                instructions.get("instruction 1", ""),
                instructions.get("instruction 2", ""),
                instructions.get("instruction 3", ""),
                instructions.get("instruction 4", ""),
            ]

            output_entry = entry.copy()
            output_entry.update({
                "select_ins": select_ins,
                "instruction": synthesized_instructions[0],
                "instructions": synthesized_instructions,
            })


            save_results(main_results_path, output_entry)
            processed_count += 1
            elapsed_time = time.time() - start_time
            print(f"Processed {processed_count + skipped_count + failed_count}/{total_entries}: {entry_key} (Time: {elapsed_time:.2f}s)")

        except Exception as e:
            failed_count += 1
            print(f"Failed to generate instructions for {entry_key}: {e}")
            continue

    print(f"=" * 80)
    print(f"Processing completed!")
    print(f"Total: {total_entries}")
    print(f"Processed in this run: {processed_count}")
    print(f"Already processed: {skipped_count + failed_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to: {main_results_path}")
    print(f"=" * 80)