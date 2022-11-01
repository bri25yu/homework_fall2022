#!/bin/bash

python cs285/scripts/run_hw4_mbpo.py \
    --exp_name q6_cheetah_rlenl0 \
    --env_name 'cheetah-cs285-v0' \
    --add_sl_noise \
    --num_agent_train_steps_per_iter 1500 \
    --batch_size_initial 5000 \
    --batch_size 5000 \
    --n_iter 10 \
    --video_log_freq -1 \
    --sac_discount 0.99 \
    --sac_n_layers 2 \
    --sac_size 256 \
    --sac_batch_size 1500 \
    --sac_learning_rate 0.0003 \
    --sac_init_temperature 0.1 \
    --sac_n_iter 1000 \
    --mbpo_rollout_length 0

python cs285/scripts/run_hw4_mbpo.py \
    --exp_name q6_cheetah_rlen1 \
    --env_name 'cheetah-cs285-v0' \
    --add_sl_noise \
    --num_agent_train_steps_per_iter 1500 \
    --batch_size_initial 5000 \
    --batch_size 5000 \
    --n_iter 10 \
    --video_log_freq -1 \
    --sac_discount 0.99 \
    --sac_n_layers 2 \
    --sac_size 256 \
    --sac_batch_size 1500 \
    --sac_learning_rate 0.0003 \
    --sac_init_temperature 0.1 \
    --sac_n_iter 5000 \
    --mbpo_rollout_length 1

# Rewards around 1000 or higher with a rollout length of 10
# Takes around 100 minutes on GPU
python cs285/scripts/run_hw4_mbpo.py \
    --exp_name q6_cheetah_rlen10 \
    --env_name 'cheetah-cs285-v0' \
    --add_sl_noise \
    --num_agent_train_steps_per_iter 1500 \
    --batch_size_initial 5000 \
    --batch_size 5000 \
    --n_iter 10 \
    --video_log_freq -1 \
    --sac_discount 0.99 \
    --sac_n_layers 2 \
    --sac_size 256 \
    --sac_batch_size 1500 \
    --sac_learning_rate 0.0003 \
    --sac_init_temperature 0.1 \
    --sac_n_iter 5000 \
    --mbpo_rollout_length 10
