#!/bin/bash
code_path=$(realpath ../mdm)
storage_path=$(realpath ./data/large_dataset)
prompts_path=$1

mkdir $storage_path
cd $code_path

while read prompt; do 
  python -m sample.generate \
    --seed=42 \
    --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
    --text_prompt="$prompt" \
    --num_repetitions=100 \
    --output_dir=$storage_path/"${prompt// /_}"
done < $prompts_path
