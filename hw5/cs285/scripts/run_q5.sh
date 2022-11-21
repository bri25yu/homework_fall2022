#!/bin/bash

awac_lambda=
iql_expectiles=(0.5 0.6 0.7 0.8 0.9 0.95 0.99)

for iql_expectile in ${iql_expectiles[*]}
do
    # Easy, supervised
    python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
    --exp_name q5_easy_supervised_lam"$awac_lambda"_tau"$iql_expectile" --use_rnd \
    --num_exploration_steps=20000 \
    --awac_lambda="$awac_lambda" \
    --iql_expectile="$iql_expectile"

    # Easy, unsupervised
    python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
    --exp_name q5_easy_unsupervised_lam"$awac_lambda"_tau"$iql_expectile" --use_rnd \
    --unsupervised_exploration \
    --num_exploration_steps=20000 \
    --awac_lambda="$awac_lambda" \
    --iql_expectile="$iql_expectile"

    # Medium, supervised
    python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 \
    --exp_name q5_iql_medium_supervised_lam"$awac_lambda"_tau"$iql_expectile" --use_rnd \
    --num_exploration_steps=20000 \
    --awac_lambda="$awac_lambda" \
    --iql_expectile="$iql_expectile"

    # Medium, unsupervised
    python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 \
    --exp_name q5_iql_medium_unsupervised_lam"$awac_lambda"_tau"$iql_expectile" --use_rnd \
    --unsupervised_exploration \
    --num_exploration_steps=20000 \
    --awac_lambda="$awac_lambda" \
    --iql_expectile="$iql_expectile"

done
