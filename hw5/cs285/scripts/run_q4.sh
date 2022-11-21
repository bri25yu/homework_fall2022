#!/bin/bash

lambdas=(0.1 1 2 10 20 50)

for lambda in ${lambdas[*]}
do
    # Easy, unsupervised
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --exp_name q4_awac_easy_unsupervised_lam"$lambda" --use_rnd --num_exploration_steps=20000 \
    --awac_lambda="$lambda" --unsupervised_exploration

    # Easy, supervised
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --exp_name q4_awac_easy_supervised_lam"$lambda" --use_rnd --num_exploration_steps=20000 \
    --awac_lambda="$lambda"

    # Medium, unsupervised
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q4_awac_medium_unsupervised_lam{} --use_rnd --num_exploration_steps=20000 \
    --awac_lambda="$lambda" --unsupervised_exploration

    # Medium, supervised
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q4_awac_medium_supervised_lam"$lambda" --use_rnd --num_exploration_steps=20000 \
    --awac_lambda="$lambda"

done
