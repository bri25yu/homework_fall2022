#!/bin/bash

env="PointmassEasy-v0"

python cs285/scripts/run_hw5_expl.py --env_name $env \
--unsupervised_exploration --use_custom_exploration --exp_name test_q1_alg
