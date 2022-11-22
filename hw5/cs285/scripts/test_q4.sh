#!/bin/bash

lambda=0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam"$lambda" --unsupervised_exploration --use_rnd --num_exploration_steps=20000 \
--awac_lambda="$lambda" --unsupervised_exploration
