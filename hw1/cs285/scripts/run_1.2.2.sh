exp_names=("HalfCheetah" "Hopper" "Walker2d")
for exp_name in ${exp_names[*]}
do
    python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/$exp_name.pkl \
        --env_name $exp_name-v4 \
        --exp_name bc_$exp_name \
        --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
        --video_log_freq -1
done
