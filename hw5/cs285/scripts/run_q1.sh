#!/bin/bash

env="PointmassEasy-v0"
python cs285/scripts/run_hw5_expl.py --env_name $env --use_rnd \
--unsupervised_exploration --exp_name q1_env1_rnd
python cs285/scripts/run_hw5_expl.py --env_name $env \
--unsupervised_exploration --exp_name q1_env1_random

env="PointmassMedium-v0"
python cs285/scripts/run_hw5_expl.py --env_name $env --use_rnd \
--unsupervised_exploration --exp_name q1_env2_rnd
python cs285/scripts/run_hw5_expl.py --env_name $env \
--unsupervised_exploration --exp_name q1_env2_random
