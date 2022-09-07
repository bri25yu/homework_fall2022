# Eval_AverageReturn : 4694.908203125
# Eval_StdReturn : 111.36351013183594
# Train_AverageReturn : 4778.55078125
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 \
    --exp_name dagger_ant \
    --eval_batch_size 100000 \
    --n_iter 10 \
    --do_dagger \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1
