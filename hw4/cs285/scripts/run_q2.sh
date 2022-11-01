#!/bin/bash

# Train AverageReturn should be around -160 and EvalAverageReturn should be around -70 to -50
python cs285/scripts/run_hw4_mb.py \
    --exp_name q2_obstacles_singleiteration \
    --env_name obstacles-cs285-v0 \
    --add_sl_noise \
    --num_agent_train_steps_per_iter 20 \
    --n_iter 1 \
    --batch_size_initial 5000 \
    --batch_size 1000 \
    --mpc_horizon 10 \
    --video_log_freq -1 \
    --mpc_action_sampling_strategy 'random'
