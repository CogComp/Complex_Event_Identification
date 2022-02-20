#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -u PR_attn.py --task_name ce  --do_lower_case --num_train_epochs 12 --output_file './outputs/complex_event_predictions.txt' --model_dir './models/'  --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 1e-6 --max_seq_length 140 --seed 42 --kshot 3 --beta_sampling_times 1 --train_file './data/complex/train_data.txt' --dev_file './data/complex/dev_data.txt' --test_file './data/complex/test_data.txt' --model_path '' --do_train


