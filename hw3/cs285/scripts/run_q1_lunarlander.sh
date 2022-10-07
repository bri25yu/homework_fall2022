# Should achieve around 150 reward after 350k timesteps with considerable variation
# Without the double-Q trick the average return often decreases after reaching 150
python cs285/scripts/run_hw3_dqn.py \
    --env_name LunarLander-v3 \
    --exp_name q1
