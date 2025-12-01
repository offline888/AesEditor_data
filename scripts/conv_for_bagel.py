
'''
{"pair_id": "pexels-photo-1236701_pexels-photo-1236701_0", "image_name": "pexels-photo-1236701", "img_ref": "data/HQ/part0/pexels-photo-1236701.jpeg", "img_lq": "data/LQ_M/pexels-photo-1236701_0.jpeg", "distortion_classes": ["temperature_warm", "brighten"], "distortion_names": ["temperature_warm_LAB", "brightness_brighten_shift_RGB"], "severities": [1, 4], "instruction_1": "Make the whole scene feel cooler and dimmer, like evening instead of golden hour. Lower the warmth and bring back some shadows to balance out the glow.", "instruction_2": "Douse that intense orange glow and cool down the whole picture. Also, turn down the brightness so it doesn’t feel so harsh and overexposed.", "instruction_3": "Apply a cool-toned filter to neutralize the excessive warmth, then reduce overall exposure to restore natural contrast and depth.", "instruction_4": "Adjust the image to feel less saturated and brighter, then cool the color temperature to bring back a more balanced, subdued mood."}

to

{"raw": "data/LQ_M/pexels-photo-1236701_0.jpeg", "target": "data/HQ/part0/pexels-photo-1236701.jpeg", "type": "single", "source": "PPR10k", "pair_id": "data/LQ_M/pexels-photo-1236701_0.jpeg->data/HQ/part0/pexels-photo-1236701.jpeg", "instructions": ["Make the whole scene feel cooler and dimmer, like evening instead of golden hour. Lower the warmth and bring back some shadows to balance out the glow.", "Douse that intense orange glow and cool down the whole picture. Also, turn down the brightness so it doesn’t feel so harsh and overexposed.", "Apply a cool-toned filter to neutralize the excessive warmth, then reduce overall exposure to restore natural contrast and depth.", "Adjust the image to feel less saturated and brighter, then cool the color temperature to bring back a more balanced, subdued mood."]}
'''

import json
import os
from tqdm import tqdm

in_jsons = [
    "data/sd_instructions_part0.jsonl",
    "data/sd_instructions_part1.jsonl",
    "data/sd_instructions_part2.jsonl",
    "data/sd_instructions_part3.jsonl",
    "data/md_instructions_part0.jsonl",
    "data/md_instructions_part1.jsonl",
    "data/md_instructions_part2.jsonl",
    "data/md_instructions_part3.jsonl",
]
test_json = "data/pexels_test.jsonl"
train_json = "data/pexels_train.jsonl"


with open(train_json, 'w') as train_out:
    with open(test_json, 'w') as test_out:
        for in_json in in_jsons:
            if not os.path.exists(in_json):
                print(f"{in_json} not found")
                continue
            print(f"processing {in_json}")
            with open(in_json, 'r') as f:
                for i, line in tqdm(enumerate(f), desc=f"processing {in_json}"):
                    data = json.loads(line)
                    img_lq = data["img_lq"].replace("data/", "")
                    img_ref = data["img_ref"].replace("data/", "")
                    write_line = json.dumps({
                        "raw": img_lq,
                        "target": img_ref,
                        "type": "single" if '/sd_' in in_json else "multi",
                        "source": "Pexels.com",
                        "pair_id": f"{img_lq}->{img_ref}",
                        "instruction": data["instruction_1"],
                        "instructions": [data["instruction_1"], data["instruction_2"], data["instruction_3"], data["instruction_4"]],
                        "distortion_class": data.get("distortion_class", None) or data["distortion_classes"],
                        "distortion_name": data.get("distortion_name", None) or data["distortion_names"],
                        "severity": data.get("severity", None) or data["severities"],
                    }, ensure_ascii=False) + "\n"
                    if i < 125:
                        test_out.write(write_line)
                    else:
                        train_out.write(write_line)