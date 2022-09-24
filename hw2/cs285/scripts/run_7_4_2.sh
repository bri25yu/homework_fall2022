batch_size=
learning_rate=

python cs285/scripts/run_hw2.py \
    --exp_name q4_b${batch_size}_r${learning_rate} \
    --env_name HalfCheetah-v4 \
    --multiprocess_gym_envs 10 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $batch_size \
    -lr $learning_rate

python cs285/scripts/run_hw2.py \
    --exp_name q4_b${batch_size}_r${learning_rate}_rtg \
    --env_name HalfCheetah-v4 \
    --multiprocess_gym_envs 10 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $batch_size \
    -lr $learning_rate \
    -rtg

python cs285/scripts/run_hw2.py \
    --exp_name q4_b${batch_size}_r${learning_rate}_nnbaseline \
    --env_name HalfCheetah-v4 \
    --multiprocess_gym_envs 10 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $batch_size \
    -lr $learning_rate \
    --nn_baseline

python cs285/scripts/run_hw2.py \
    --exp_name q4_b${batch_size}_r${learning_rate}_rtg_nnbaseline \
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
