#!/bin/bash

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --exploit_rew_shift 1 --exploit_rew_scale 100 --exp_name q3_medium_dqn_scaled
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --exploit_rew_shift 1 --exploit_rew_scale 100 --exp_name q3_hard_dqn_scaled
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql
