
part=0

python download/download_images.py --part 4000-${part} --output_dir data/HQ/part${part}


python scripts/add_distortion_sd.py --reference_dir data/HQ/part${part} --distortion_dir data/LQ_S --json_path data/sd_part${part}.json &

python scripts/add_distortion_md.py --reference_dir data/HQ/part${part} --distortion_dir data/LQ_M --json_path data/md_part${part}.json

wait

# pip install transformers==4.57 # >=4.57
CUDA_VISIBLE_DEVICES=$((${part}*2)) python scripts/gen_instruction_sd.py --meta_json data/sd_part${part}.json --save_json data/sd_instructions_part${part}.jsonl --model-path Qwen/Qwen3-VL-8B-Instruct &

CUDA_VISIBLE_DEVICES=$((${part}*2+1)) python scripts/gen_instruction_md.py --meta_json data/md_part${part}.json --save_json data/md_instructions_part${part}.jsonl --model-path Qwen/Qwen3-VL-8B-Instruct