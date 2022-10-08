ntu=
ngsptu=

# Around 20 iters should be 100 return
# Around 100 iters should be 1000 return
python cs285/scripts/run_hw3_actor_critic.py \
    --env_name InvertedPendulum-v4 \
    --exp_name q5_${ntu}_${ngsptu} \
    --ep_len 1000 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 64 \
    -b 5000 \
    -lr 0.01 \
    -ntu $ntu \
    -ngsptu $ngsptu

# Around 20 iters should be > -40 return
# Around 150 iters should be 150 return
python cs285/scripts/run_hw3_actor_critic.py \
    --env_name HalfCheetah-v4 \
    --exp_name q5_${ntu}_${ngsptu} \
    --ep_len 150 \
    --discount 0.90 \
    --scalar_log_freq 1 \
    -n 150 \
    -l 2 \
    -s 32 \
    -b 30000 \
    -eb 1500 \
    -lr 0.02 \
    -ntu $ntu \
    -ngsptu $ngsptu
