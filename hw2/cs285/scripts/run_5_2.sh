batch_sizes=(10000 5000 1000 500)
learning_rates=(2e-3 5e-3 8e-3)


for batch_size in ${batch_sizes[*]}
do
    for learning_rate in ${learning_rates[*]}
    do
        python cs285/scripts/run_hw2.py \
            --exp_name q2_b${batch_size}_r${learning_rate} \
            --env_name InvertedPendulum-v4 \
            --multiprocess_gym_envs 10 \
            --ep_len 1000 \
            --discount 0.9 \
            -n 100 \
            -l 2 \
            -s 64 \
            -b $batch_size \
            -lr $learning_rate \
            -rtg
    done
done
