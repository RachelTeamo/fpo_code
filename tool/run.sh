
# python loss_gradient_analyzer.py \
#     --weight_path /path/to/weights.pth \
#     --data_path /home/user/code/zty/datasets/  \
#     --num_samples 64 \
#     --output_dir ./sample_analysis_result_top50_64samples_random_seed_42 \
#     --random_seed 42 


# python loss_gradient_analyzer.py \
#     --weight_path /path/to/weights.pth \
#     --data_path /home/user/code/zty/datasets/  \
#     --num_samples 64 \
#     --output_dir ./sample_analysis_result_top50_64samples_random_seed_42_2 \
#     --random_seed 42 

# dit-b
python loss_gradient_analyzer.py \
    --weight_path /path/to/weights.pth \
    --data_path /home/user/code/zty/datasets/  \
    --num_samples 32 \
    --output_dir ./dit_b_sample_analysis_result_top50_32samples_random_seed_42 \
    --random_seed 42 \
    --model_config ./config/DiT-B.yaml

# dit-l
# python loss_gradient_analyzer.py \
#     --weight_path /path/to/weights.pth \
#     --data_path /home/user/code/zty/datasets/  \
#     --num_samples 16 \
#     --output_dir ./dit_l_sample_analysis_result_top50_16samples_random_seed_42 \
#     --random_seed 42 \
#     --model_config ./config/DiT-L.yaml
