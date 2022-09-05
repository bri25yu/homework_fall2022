# Eval_AverageReturn : 4003.580078125
# Eval_StdReturn : 106.2594985961914
# Train_AverageReturn : 4205.7783203125
exp_name="HalfCheetah"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1

# Eval_AverageReturn : 1627.7149658203125
# Eval_StdReturn : 324.4638977050781
# Train_AverageReturn : 3772.67041015625
# Tuned over 8e-4, 1e-3, 2e-3, 3e-3, best value at 1e-3
exp_name="Hopper"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --num_agent_train_steps_per_iter 10000 \
    -lr 1e-3 \
    --loss L1Loss \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1


# Eval_AverageReturn : 4045.609375
# Eval_StdReturn : 1616.4520263671875
# Train_AverageReturn : 5566.845703125
# Tuned over 1e-3, 2e-3, 3e-3, 5e-3, best value at 3e-3
exp_name="Walker2d"
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --eval_batch_size 100000 \
    --num_agent_train_steps_per_iter 10000 \
    -lr 3e-3 \
    --loss L1Loss \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1
