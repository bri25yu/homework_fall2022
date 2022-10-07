seeds=(1 2 3)

for seed in ${seeds[*]}
do
    python cs285/scripts/run_hw3_dqn.py \
        --env_name LunarLander-v3 \
        --exp_name q2_dqn_$seed \
        --seed $seed
done

for seed in ${seeds[*]}
do
    python cs285/scripts/run_hw3_dqn.py \
        --env_name LunarLander-v3 \
        --exp_name q2_doubledqn_$seed \
        --double_q \
        --seed $seed
done
