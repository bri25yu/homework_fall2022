# Unfortunately we can't use multiple envs for this as I haven't tested the vectorenv code with action wrappers ://

gae_lambdas=(0 0.95 0.98 0.99 1)

for gae_lambda in ${gae_lambdas[*]}
do
    python cs285/scripts/run_hw2.py \
        --exp_name q5_b2000_r0.001_lambda${gae_lambda} \
        --env_name Hopper-v4 \
        --ep_len 1000 \
        --discount 0.99 \
        -n 300 \
        -l 2 \
        -s 32 \
        -b 2000 \
        -lr 0.001 \
        --reward_to_go \
        --nn_baseline \
        --action_noise_std 0.5 \
        --gae_lambda $gae_lambda
done
