# Eval_AverageReturn : 4100.78564453125
# Eval_StdReturn : 91.00215911865234
# Train_AverageReturn : 4015.66259765625
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

# Eval_AverageReturn : 3716.4033203125
# Eval_StdReturn : 3.8703887462615967
# Train_AverageReturn : 3718.23876953125
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

# Early stopping was performed at n_iter=8
# Eval_AverageReturn : 5428.83203125
# Eval_StdReturn : 37.54182815551758
# Train_AverageReturn : 5396.89013671875
exp_name="Walker2d"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --n_iter 8 \
    --do_dagger \
    --num_agent_train_steps_per_iter 10000 \
    -lr 1e-3 \
    --loss L1Loss \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1
