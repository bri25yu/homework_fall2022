# The best config of the 4 below should be around 200 return
python cs285/scripts/run_hw3_actor_critic.py \
    --env_name CartPole-v0 \
    --exp_name q4_ac_1_1 \
    -n 100 \
    -b 1000 \
    -ntu 1 \
    -ngsptu 1

python cs285/scripts/run_hw3_actor_critic.py \
    --env_name CartPole-v0 \
    --exp_name q4_100_1 \
    -n 100 \
    -b 1000 \
    -ntu 100 \
    -ngsptu 1

python cs285/scripts/run_hw3_actor_critic.py \
    --env_name CartPole-v0 \
    --exp_name q4_1_100 \
    -n 100 \
    -b 1000 \
    -ntu 1 \
    -ngsptu 100

python cs285/scripts/run_hw3_actor_critic.py \
    --env_name CartPole-v0 \
    --exp_name q4_10_10 \
    -n 100 \
    -b 1000 \
    -ntu 10 \
    -ngsptu 10
