#!/bin/bash

awac_lambda=0.1
iql_expectile=0.5

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
--exp_name q5_easy_supervised_lam"$awac_lambda"_tau"$iql_expectile" --use_rnd \
--num_exploration_steps=20000 \
--awac_lambda="$awac_lambda" \
--iql_expectile="$iql_expectile"
