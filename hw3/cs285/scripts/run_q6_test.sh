# After 10000 steps, return should be near or above 100
# After 20000 steps, return should reach 1000
# Target 1000 eval average return under 100000 steps
python cs285/scripts/run_hw3_sac.py \
    --env_name InvertedPendulum-v4 \
    ----exp_name q6a_sac_InvertedPendulum \
    --ep_len 1000 \
    --discount 0.99 \
    --scalar_log_freq 1000 \
    -n 100000 \
    -l 2 \
    -s 256 \
    -b 1000 \
    -eb 2000 \
    -lr 0.0003 \
    --init_temperature 0.1 \
    --seed 1
