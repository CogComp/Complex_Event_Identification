#!/bin/bash
MODEL_PATH='models/attn_47_0.5330089978613562.pt'
# get dev pairwise prediction
CUDA_VISIBLE_DEVICES=0 python -u PR_attn.py --task_name ce  --do_lower_case --num_train_epochs 12 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 1e-6 --max_seq_length 140 --seed 42 --kshot 3 --beta_sampling_times 1 --train_file './data/complex/train_data.txt' --dev_file './data/complex/dev_data.txt' --test_file './data/complex/dev_data.txt' --model_path $MODEL_PATH --output_file './outputs/dev_complex_event_predictions.txt' --model_dir './models/'
# get test pairwise prediction
CUDA_VISIBLE_DEVICES=0 python -u PR_attn.py --task_name ce  --do_lower_case --num_train_epochs 12  --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 1e-6 --max_seq_length 140 --seed 42 --kshot 3 --beta_sampling_times 1 --train_file './data/complex/train_data.txt' --dev_file './data/complex/dev_data.txt' --test_file './data/complex/test_data.txt' --model_path $MODEL_PATH --output_file './outputs/test_complex_event_predictions.txt' --model_dir './models/'
# clustering step
python clustering.py
