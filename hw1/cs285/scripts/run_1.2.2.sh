exp_name="HalfCheetah"
n_layers=2
size=64
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --n_iter 1 \
    --n_layers $n_layers \
    --size $size \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1

# Best is 1471 at 2/1024/10000 over 1e-5, 2e-5, 3e-5 at 3e-5
exp_name="Hopper"
n_layers=2
size=4096
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/$exp_name.pkl \
    --env_name $exp_name-v4 \
    --exp_name bc_$exp_name \
    --n_iter 1 \
    --n_layers $n_layers \
    --size $size \
    --num_agent_train_steps_per_iter 10000 \
    -lr 2e-5 \
    --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
    --video_log_freq -1

# exp_name="Walker2d"
# n_layers=2
# size=64
# python cs285/scripts/run_hw1.py \
#     --expert_policy_file cs285/policies/experts/$exp_name.pkl \
#     --env_name $exp_name-v4 \
#     --exp_name bc_$exp_name \
#     --n_iter 1 \
#     --n_layers $n_layers \
#     --size $size \
#     --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
#     --video_log_freq -1
