
'''
{"pair_id": "pexels-photo-1236701_pexels-photo-1236701_0", "image_name": "pexels-photo-1236701", "img_ref": "data/HQ/part0/pexels-photo-1236701.jpeg", "img_lq": "data/LQ_M/pexels-photo-1236701_0.jpeg", "distortion_classes": ["temperature_warm", "brighten"], "distortion_names": ["temperature_warm_LAB", "brightness_brighten_shift_RGB"], "severities": [1, 4], "instruction_1": "Make the whole scene feel cooler and dimmer, like evening instead of golden hour. Lower the warmth and bring back some shadows to balance out the glow.", "instruction_2": "Douse that intense orange glow and cool down the whole picture. Also, turn down the brightness so it doesn’t feel so harsh and overexposed.", "instruction_3": "Apply a cool-toned filter to neutralize the excessive warmth, then reduce overall exposure to restore natural contrast and depth.", "instruction_4": "Adjust the image to feel less saturated and brighter, then cool the color temperature to bring back a more balanced, subdued mood."}

to

{"raw": "data/LQ_M/pexels-photo-1236701_0.jpeg", "target": "data/HQ/part0/pexels-photo-1236701.jpeg", "type": "single", "source": "PPR10k", "pair_id": "data/LQ_M/pexels-photo-1236701_0.jpeg->data/HQ/part0/pexels-photo-1236701.jpeg", "instructions": ["Make the whole scene feel cooler and dimmer, like evening instead of golden hour. Lower the warmth and bring back some shadows to balance out the glow.", "Douse that intense orange glow and cool down the whole picture. Also, turn down the brightness so it doesn’t feel so harsh and overexposed.", "Apply a cool-toned filter to neutralize the excessive warmth, then reduce overall exposure to restore natural contrast and depth.", "Adjust the image to feel less saturated and brighter, then cool the color temperature to bring back a more balanced, subdued mood."]}
'''

import json
import os

in_json = "data/md_instructions_part0.jsonl"
out_json = "data/md_instructions_part0_bagel.jsonl"


with open(in_json, 'r') as f:
    with open(out_json, 'w') as f_out:
        for line in f:
            data = json.loads(line)
            f_out.write(json.dumps({
                "raw": data["img_lq"],
                "target": data["img_ref"],
                "type": "single" if '/sd_' in data["img_lq"] else "multi",
                "source": "Pexels.com",
                "pair_id": f"{data['img_lq']}->{data['img_ref']}",
                "instruction": data["instruction_1"],
                "instructions": [data["instruction_1"], data["instruction_2"], data["instruction_3"], data["instruction_4"]],
                "distortion_class": data.get("distortion_class", None) or data["distortion_classes"],
                "distortion_name": data.get("distortion_name", None) or data["distortion_names"],
                "severity": data.get("severity", None) or data["severities"],
            }) + "\n")