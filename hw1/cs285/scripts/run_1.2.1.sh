# Eval_AverageReturn : 4689.48193359375
# Eval_StdReturn : 421.529052734375
# Train_AverageReturn : 4713.6533203125
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 \
    --exp_name bc_ant \
    --eval_batch_size 100000 \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1
