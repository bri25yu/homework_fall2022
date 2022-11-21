#!/bin/bash

python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \
--unsupervised_exploration --use_custom_exploration --exp_name q1_alg_med
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
--unsupervised_exploration --use_custom_exploration --exp_name q1_alg_med
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 \
--unsupervised_exploration --use_custom_exploration --exp_name q1_alg_hard
