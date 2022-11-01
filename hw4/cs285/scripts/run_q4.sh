#!/bin/bash

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_horizon5 \
    --env_name reacher-cs285-v0 \
    --add_sl_noise \
    --mpc_horizon 5 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_horizon15 \
    --env_name reacher-cs285-v0 \
    --add_sl_noise \
    --mpc_horizon 15 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_horizon30 \
    --env_name reacher-cs285-v0 \
    --add_sl_noise \
    --mpc_horizon 30 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_numseq100 \
    --env_name reacher-cs285-v0 \
    --add_sl_noise \
    --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --mpc_num_action_sequences 100 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_numseq1000 \
    --env_name reacher-cs285-v0 \
    --add_sl_noise \
    --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_num_action_sequences 1000 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_ensemble1 \
    --env_name reacher-cs285-v0 \
    --ensemble_size 1 \
    --add_sl_noise \
    --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_ensemble3 \
    --env_name reacher-cs285-v0 \
    --ensemble_size 3 \
    --add_sl_noise \
    --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py \
    --exp_name q4_reacher_ensemble5 \
    --env_name reacher-cs285-v0 \
    --ensemble_size 5 \
    --add_sl_noise \
    --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 \
    --n_iter 15 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'
