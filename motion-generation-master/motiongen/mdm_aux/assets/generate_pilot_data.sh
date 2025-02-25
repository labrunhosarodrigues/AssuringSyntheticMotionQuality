#!/bin/bash
code_path=$(realpath ../mdm)
storage_path=$(realpath ./data/$(date +%Y_%m_%dT%H_%M_%S))
prompts_path=$(realpath ./pilot_generation/prompts.txt)

mkdir $storage_path
cd $code_path

python -m sample.generate --seed=42 --model_path ./save/humanml_enc_512_50steps/model000750000.pt --input_text=$prompts_path --num_repetitions=10 --output_dir=$storage_path
