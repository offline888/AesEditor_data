
mkdir data
cd data
python ../download/download_images.sh
mv imgs HQ
cd ..


python scripts/add_distortion_sd.py --reference_dir data/HQ --distortion_dir data/LQ_S --json_path data/sd.json

python scripts/add_distortion_md.py --reference_dir data/HQ --distortion_dir data/LQ_M --json_path data/md.json



# pip install transformers==4.57 # >=4.57
python scripts/gen_instruction_sd.py --meta_json data/sd.json --save_json data/sd_instructions.jsonl --model-path Qwen/Qwen3-VL-8B-Instruct

python scripts/gen_instruction_md.py --meta_json data/md.json --save_json data/md_instructions.jsonl --model-path Qwen/Qwen3-VL-8B-Instruct