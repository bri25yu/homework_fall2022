exp_name="HalfCheetah"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --n_iter 10 \
    --do_dagger \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1

# Best is 1732 at 2/64/10000/L1Loss over 1e-3, 2e-3, 3e-3 at 2e-3
exp_name="Hopper"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --n_iter 10 \
    --do_dagger \
    --num_agent_train_steps_per_iter 10000 \
    -lr 2e-3 \
    --loss L1Loss \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1

# Best is 5272 at 2/64/10000/L1Loss over 1e-3, 2e-3, 3e-3 at 1e-3
exp_name="Walker2d"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --n_iter 10 \
    --do_dagger \
    --num_agent_train_steps_per_iter 10000 \
    -lr 1e-3 \
    --loss L1Loss \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1