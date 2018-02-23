#!/bin/bash

export PYTHONPATH="$(pwd)"
export TF_MIN_GPU_MULTIPROCESSOR_COUNT="3"
export CUDA_VISIBLE_DEVICES="1"

python src/cifar10/main.py \
  --data_format="NCHW" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="/home/hyhieu/data/cifar10" \
  --output_dir="/home/hyhieu/Desktop/cifar10/outputs" \
  --batch_size=64 \
  --num_epochs=310 \
  --log_every=1 \
  --eval_every_epochs=1 \
  --child_num_layers=6 \
  --child_out_filters=48 \
  --child_l2_reg=1e-4 \
  --child_num_branches=4 \
  --child_num_cell_layers=5 \
  --child_keep_prob=0.50 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.01 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_num_train_steps=25 \
  --controller_lr=1e-3 \
  --controller_tanh=2.5 \
  --controller_temperature=5.0 \
  --controller_skip_target=0.4 \
  --controller_skip_rate=0.50 \
  "$@"

