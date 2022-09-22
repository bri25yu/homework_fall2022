# All the following shift the advantages calculated to ensure a non-negative loss
# All the following experiments don't use reward-to-go and don't standardize the advantages

# Cartpole env
python cs285/scripts/run_hw2.py \
    --exp_name shift_advantages_cartpole \
    --env_name CartPole-v0 \
    -n 100 \
    -b 1000 \
    -dsa \
    --shift_advantages

# InvertedPendulum env
python cs285/scripts/run_hw2.py \
    --exp_name shift_advantages_InvertedPendulum \
    --env_name InvertedPendulum-v4 \
    --ep_len 1000 \
    --discount 0.9 \
    -n 100 \
    -l 2 \
    -s 64 \
    -b 10000 \
    --shift_advantages

# LunarLander env
python cs285/scripts/run_hw2.py \
    --exp_name shift_advantages_LunarLanderContinuous \
    --env_name LunarLanderContinuous-v2 \
    --ep_len 1000 \
    --discount 0.99 \
    -n 100 \
    -l 2 \
    -s 64 \
    -b 40000 \
    -lr 0.005 \
    --shift_advantages

# HalfCheetah env
python cs285/scripts/run_hw2.py \
    --exp_name shift_advantages_HalfCheetah \
    --env_name HalfCheetah-v4 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b 30000 \
    --shift_advantages
