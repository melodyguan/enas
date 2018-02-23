#!/bin/bash

export PYTHONPATH="$(pwd)"
export TF_MIN_GPU_MULTIPROCESSOR_COUNT="8"
export CUDA_VISIBLE_DEVICES="0"

python src/neural_mt/main.py \
  --search_for="enas" \
  --reset_output_dir \
  --data_path="/home/hyhieu/data/wmt/en_de" \
  --source_lang="en" \
  --target_lang="de" \
  --train_prefix="train_bpe" \
  --valid_prefix="newstest2013_bpe" \
  --vocab_prefix="vocab_bpe_32000" \
  --output_dir="/home/hyhieu/Desktop/neural_mt/outputs" \
  --log_every=1 \
  --eval_every_epochs=1

