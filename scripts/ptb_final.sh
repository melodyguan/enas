#!/bin/bash

export PYTHONPATH="$(pwd)"

fixed_arc="0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"

python src/ptb/main.py \
  --search_for="enas" \
  --reset_output_dir \
  --data_path="data/ptb/ptb.pkl" \
  --output_dir="outputs" \
  --batch_size=64 \
  --child_bptt_steps=35 \
  --num_epochs=2000 \
  --child_fixed_arc="${fixed_arc}" \
  --child_rhn_depth=12 \
  --child_num_layers=1 \
  --child_lstm_hidden_size=748 \
  --child_lstm_e_keep=0.79 \
  --child_lstm_x_keep=0.25 \
  --child_lstm_h_keep=0.75 \
  --child_lstm_o_keep=0.24 \
  --nochild_lstm_e_skip \
  --child_grad_bound=0.25 \
  --child_lr=20.0 \
  --child_rnn_slowness_reg=1e-3 \
  --child_l2_reg=5e-7 \
  --child_lr_dec_start=14 \
  --child_lr_dec_every=1 \
  --child_lr_dec_rate=0.9991 \
  --child_lr_dec_min=0.001 \
  --child_optim_algo="sgd" \
  --log_every=50 \
  --nocontroller_training \
  --controller_selection_threshold=5 \
  --controller_train_every=1 \
  --controller_lr=0.001 \
  --controller_sync_replicas \
  --controller_train_steps=100 \
  --controller_num_aggregate=10 \
  --controller_tanh_constant=3.0 \
  --controller_temperature=2.0 \
  --controller_entropy_weight=0.0001 \
  --eval_every_epochs=1

