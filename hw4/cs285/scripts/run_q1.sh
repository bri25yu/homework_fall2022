#!/bin/bash

# Loss should go below 0.2 by iteration 500
python cs285/scripts/run_hw4_mb.py \
    --exp_name q1_cheetah_n500_arch1x32 \
    --env_name cheetah-cs285-v0 \
    --add_sl_noise \
    --n_iter 1 \
    --batch_size_initial 20000 \
    --num_agent_train_steps_per_iter 500 \
    --n_layers 1 \
    --size 32 \
    --scalar_log_freq -1 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q1_cheetah_n5_arch2x250 \
    --env_name cheetah-cs285-v0 \
    --add_sl_noise \
    --n_iter 1 \
    --batch_size_initial 20000 \
    --num_agent_train_steps_per_iter 5 \
    --n_layers 2 \
    --size 250 \
    --scalar_log_freq -1 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q1_cheetah_n500_arch2x250 \
    --env_name cheetah-cs285-v0 \
    --add_sl_noise \
    --n_iter 1 \
    --batch_size_initial 20000 \
    --num_agent_train_steps_per_iter 500 \
    --n_layers 2 \
    --size 250 \
    --scalar_log_freq -1 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'
