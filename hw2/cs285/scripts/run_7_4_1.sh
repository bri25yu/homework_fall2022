batch_sizes=(10000 30000 50000)
learning_rates=(5e-3 1e-2 2e-2)


for batch_size in ${batch_sizes[*]}
do
    for learning_rate in ${learning_rates[*]}
    do
        python cs285/scripts/run_hw2.py \
            --exp_name q4_search_b${batch_size}_lr${learning_rate}_rtg_nnbaseline \
            --env_name HalfCheetah-v4 \
            --multiprocess_gym_envs 10 \
            --ep_len 150 \
            --discount 0.95 \
            -n 100 \
            -l 2 \
            -s 32 \
            -b $batch_size \
            -lr $learning_rate \
            -rtg \
            --nn_baseline

    done
done
