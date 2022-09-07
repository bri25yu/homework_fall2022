exp_name="Hopper"

losses=(
    "MSELoss"
    "HuberLoss"
    "L1Loss"
)
learning_rates=(
    1e-3
    2e-3
    3e-3
)

for loss in ${losses[*]}
do
    for learning_rate in ${learning_rates[*]}
    do
        python cs285/scripts/run_hw1.py \
            --expert_policy_file cs285/policies/experts/$exp_name.pkl \
            --env_name $exp_name-v4 \
            --exp_name q1_bc_$exp_name_$loss_$learning_rate \
            --eval_batch_size 100000 \
            --num_agent_train_steps_per_iter 10000 \
            -lr $learning_rate \
            --loss $loss \
            --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl \
            --video_log_freq -1
    done
done
