#!/bin/bash

env="PointmassMedium-v0"

expl_step_values=(5000 15000)

for expl_step_value in ${expl_step_values[*]}
do
    python cs285/scripts/run_hw5_expl.py --env_name $env --use_rnd \
    --num_exploration_steps="$expl_step_value" --exploit_rew_shift 1 --exploit_rew_scale 100 --offline_exploitation --cql_alpha=0.1 \
    --unsupervised_exploration --exp_name q2_cql_numsteps_"$expl_step_value"

    python cs285/scripts/run_hw5_expl.py --env_name $env --use_rnd \
    --num_exploration_steps="$expl_step_value" --exploit_rew_shift 1 --exploit_rew_scale 100 --offline_exploitation --cql_alpha=0.0 \
    --unsupervised_exploration --exp_name q2_dqn_numsteps_"$expl_step_value"
done
