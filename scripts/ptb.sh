#!/bin/bash

export PYTHONPATH="$(pwd)"
export TF_MIN_GPU_MULTIPROCESSOR_COUNT=3
export CUDA_VISIBLE_DEVICES="1"

python src/ptb/main.py \
  --search_for="enas" \
  --reset_output_dir \
  --data_path="/home/hyhieu/data/ptb/ptb.pkl" \
  --output_dir="/home/hyhieu/Desktop/ptb/outputs" \
  --batch_size=64 \
  --child_bptt_steps=35 \
  --num_epochs=50 \
  --child_rhn_depth=12 \
  --child_num_layers=1 \
  --child_lstm_hidden_size=20 \
  --child_lstm_e_keep=1.0 \
  --child_lstm_x_keep=1.0 \
  --child_lstm_h_keep=1.0 \
  --child_lstm_o_keep=1.0 \
  --child_lstm_e_skip \
  --child_grad_bound=10.0 \
  --child_lr=0.2 \
  --child_lr_dec_start=14 \
  --child_lr_dec_every=2 \
  --child_lr_dec_rate=0.9 \
  --child_lr_dec_min=0.005 \
  --log_every=1 \
  --controller_training \
  --controller_selection_threshold=1 \
  --controller_train_every=5 \
  --controller_lr=0.01 \
  --controller_sync_replicas \
  --controller_train_steps=250 \
  --controller_num_aggregate=10 \
  --controller_skip_target=0.60 \
  --controller_skip_rate=0.1 \
  --controller_entropy_weight=0.00001 \
  --eval_every_epochs=1

