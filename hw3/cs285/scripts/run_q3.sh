seeds=(1 2 3)

# With lr scheduling, warmup_ratio=0.1
for seed in ${seeds[*]}
do
    python cs285/scripts/run_hw3_dqn.py \
        --env_name LunarLander-v3 \
        --exp_name q3_hparam1_$seed \
        --seed $seed \
        --use_learning_rate_scheduler \
        --warmup_ratio 0.1
done

# With AdamW
for seed in ${seeds[*]}
do
    python cs285/scripts/run_hw3_dqn.py \
        --env_name LunarLander-v3 \
        --exp_name q3_hparam2_$seed \
        --seed $seed \
        --use_adamw
done

# With both lr scheduling and AdamW
for seed in ${seeds[*]}
do
    python cs285/scripts/run_hw3_dqn.py \
        --env_name LunarLander-v3 \
        --exp_name q3_hparam3_$seed \
        --seed $seed \
        --use_learning_rate_scheduler \
        --warmup_ratio 0.1 \
        --use_adamw
done
