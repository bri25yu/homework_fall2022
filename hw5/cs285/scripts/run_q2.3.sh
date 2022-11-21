#!/bin/bash

env="PointmassMedium-v0"

cql_alphas=(0.02 0.5)

for cql_alpha in ${cql_alphas[*]}
do
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --unsupervised_exploration --offline_exploitation --cql_alpha="$cql_alpha" \
    --exp_name q2_alpha"$cql_alpha"
done
