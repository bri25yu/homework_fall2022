#!/bin/bash

# Rewards -25 to -20 (takes 40 minutes, no GPU)
python cs285/scripts/run_hw4_mb.py \
    --exp_name q3_obstacles \
    --env_name obstacles-cs285-v0 \
    --add_sl_noise \
    --num_agent_train_steps_per_iter 20 \
    --batch_size_initial 5000 \
    --batch_size 1000 \
    --mpc_horizon 10 \
    --n_iter 12 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

# Rewards -250 to -300 (takes 2-3 hours, no GPU)
python cs285/scripts/run_hw4_mb.py \
    --exp_name q3_reacher \
    --env_name reacher-cs285-v0 \
    --add_sl_noise \
    --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size_initial 5000 \
    --batch_size 5000 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

# Rewards around 250-350 for the cheetah env (takes 3-4 hours, no GPU)
python cs285/scripts/run_hw4_mb.py \
    --exp_name q3_cheetah \
    --env_name cheetah-cs285-v0 \
    --mpc_horizon 15 \
    --add_sl_noise \
    --num_agent_train_steps_per_iter 1500 \
    --batch_size_initial 5000 \
    --batch_size 5000 \
    --n_iter 20 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'
