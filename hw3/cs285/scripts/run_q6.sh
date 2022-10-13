# After 10000 steps, return should be above -40 (trending toward positive)
# After 50000 steps, return should be around 200
# Target 300 average return under 200000 steps
python cs285/scripts/run_hw3_sac.py \
    --env_name HalfCheetah-v4 --ep_len 150 \
    --exp_name q6b_sac_HalfCheetah \
    --discount 0.99 \
    --scalar_log_freq 1500 \
    -n 2000000 \
    -l 2 \
    -s 256 \
    -b 1500 \
    -tb 1500 \
    -eb 1500 \
    -lr 1e-4 \
    --init_temperature 0.1 \
    --seed 1
