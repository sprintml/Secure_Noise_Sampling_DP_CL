#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -c 20
#SBATCH --array=0


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo timestamp: ${timestamp}ro_b
start_time=$(date +%s)

#use_discrete=0 for continuous noise
#use_discrete=1 for discrete noise with gradient discretization

dataset_name='sst2'
method_type='prompt'
model_name_or_path='roberta-base'
per_device_train_batch_size=180
gradient_accumulation_steps=5
learning_rate=0.05
target_epsilon=8.0
per_sample_max_grad_norm=0.01
num_global_server_epochs=21
pre_seq_len=9
num_clients=10
max_seq_length=256

export PYTHONPATH="${PYTHONPATH}:path/to/this_repo"
path/to/env/bin/python3.8 path/to/this_repo/server.py \
--model_name_or_path ${model_name_or_path} --training_type private --task_name glue --dataset_name ${dataset_name} \
--do_train --do_eval --max_seq_length ${max_seq_length} --per_device_train_batch_size ${per_device_train_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--learning_rate ${learning_rate} --num_train_epochs 1 --pre_seq_len ${pre_seq_len} \
--global_overwrite_output_dir 0 \
--hidden_dropout_prob 0.1 --seed ${SLURM_ARRAY_TASK_ID} --save_strategy no --evaluation_strategy epoch --training_type private \
--target_epsilon ${target_epsilon} --per_sample_max_grad_norm ${per_sample_max_grad_norm} --remove_unused_columns True --label_name labels \
--privacy_engine private_transformers --output "" --num_global_server_epochs ${num_global_server_epochs} --lr_decay no \
--num_clients ${num_clients} --save_freq_aggregate 200 --method_type ${method_type} \
--global_output_dir checkpoints/${dataset_name}_${model_name_or_path}_${method_type}_private_transformer_max${max_seq_length}_use_discrete${use_discrete}_${SLURM_ARRAY_TASK_ID}/ \
--freeze_non_prompt_layers 0 --use_discrete ${use_discrete}




dataset_name='qnli'
method_type='prompt'
model_name_or_path='roberta-base'
per_device_train_batch_size=210
gradient_accumulation_steps=5
learning_rate=0.005
target_epsilon=8.0
per_sample_max_grad_norm=0.05
num_global_server_epochs=100
pre_seq_len=10
num_clients=10
max_seq_length=256

export PYTHONPATH="${PYTHONPATH}:path/to/this_repo"
path/to/env/bin/python3.8 path/to/this_repo/server.py \
--model_name_or_path ${model_name_or_path} --training_type private --task_name glue --dataset_name ${dataset_name} \
--do_train --do_eval --max_seq_length ${max_seq_length} --per_device_train_batch_size ${per_device_train_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--learning_rate ${learning_rate} --num_train_epochs 1 --pre_seq_len ${pre_seq_len} \
--global_overwrite_output_dir 0 \
--hidden_dropout_prob 0.1 --seed ${SLURM_ARRAY_TASK_ID} --save_strategy no --evaluation_strategy epoch --training_type private \
--target_epsilon ${target_epsilon} --per_sample_max_grad_norm ${per_sample_max_grad_norm} --remove_unused_columns True --label_name labels \
--privacy_engine private_transformers --output "" --num_global_server_epochs ${num_global_server_epochs} --lr_decay no \
--num_clients ${num_clients} --save_freq_aggregate 200 --method_type ${method_type} \
--global_output_dir checkpoints/${dataset_name}_${model_name_or_path}_${method_type}_private_transformer_max${max_seq_length}_use_discrete${use_discrete}_${SLURM_ARRAY_TASK_ID}/ \
--freeze_non_prompt_layers 0 --use_discrete ${use_discrete}


#
dataset_name='mnli'
method_type='prompt'
model_name_or_path='roberta-base'
per_device_train_batch_size=210
gradient_accumulation_steps=5
learning_rate=0.005
target_epsilon=8.0
per_sample_max_grad_norm=0.05
num_global_server_epochs=60
pre_seq_len=10
num_clients=10
max_seq_length=256


export PYTHONPATH="${PYTHONPATH}:path/to/this_repo"
path/to/env/bin/python3.8 path/to/this_repo/server.py \
--model_name_or_path ${model_name_or_path} --training_type private --task_name glue --dataset_name ${dataset_name} \
--do_train --do_eval --max_seq_length ${max_seq_length} --per_device_train_batch_size ${per_device_train_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--learning_rate ${learning_rate} --num_train_epochs 1 --pre_seq_len ${pre_seq_len} \
--global_overwrite_output_dir 0 \
--hidden_dropout_prob 0.1 --seed ${SLURM_ARRAY_TASK_ID} --save_strategy no --evaluation_strategy epoch --training_type private \
--target_epsilon ${target_epsilon} --per_sample_max_grad_norm ${per_sample_max_grad_norm} --remove_unused_columns True --label_name labels \
--privacy_engine private_transformers --output "" --num_global_server_epochs ${num_global_server_epochs} --lr_decay no \
--num_clients ${num_clients} --save_freq_aggregate 200 --method_type ${method_type} \
--global_output_dir checkpoints/${dataset_name}_${model_name_or_path}_${method_type}_private_transformer_max${max_seq_length}_use_discrete${use_discrete}_${SLURM_ARRAY_TASK_ID}/ \
--freeze_non_prompt_layers 0 --use_discrete ${use_discrete}


dataset_name='qqp'
method_type='prompt'
model_name_or_path='roberta-base'
per_device_train_batch_size=210
gradient_accumulation_steps=5
learning_rate=0.05
target_epsilon=8.0
per_sample_max_grad_norm=0.1
num_global_server_epochs=10
pre_seq_len=7
num_clients=10
max_seq_length=256


export PYTHONPATH="${PYTHONPATH}:path/to/this_repo"
path/to/env/bin/python3.8 path/to/this_repo/server.py \
--model_name_or_path ${model_name_or_path} --training_type private --task_name glue --dataset_name ${dataset_name} \
--do_train --do_eval --max_seq_length ${max_seq_length} --per_device_train_batch_size ${per_device_train_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--learning_rate ${learning_rate} --num_train_epochs 1 --pre_seq_len ${pre_seq_len} \
--global_overwrite_output_dir 0 \
--hidden_dropout_prob 0.1 --seed ${SLURM_ARRAY_TASK_ID} --save_strategy no --evaluation_strategy epoch --training_type private \
--target_epsilon ${target_epsilon} --per_sample_max_grad_norm ${per_sample_max_grad_norm} --remove_unused_columns True --label_name labels \
--privacy_engine private_transformers --output "" --num_global_server_epochs ${num_global_server_epochs} --lr_decay no \
--num_clients ${num_clients} --save_freq_aggregate 200 --method_type ${method_type} \
--global_output_dir checkpoints/${dataset_name}_${model_name_or_path}_${method_type}_private_transformer_max${max_seq_length}_use_discrete${use_discrete}_${SLURM_ARRAY_TASK_ID}/ \
--freeze_non_prompt_layers 0 --use_discrete ${use_discrete}





end_time=$(date +%s)
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo timestamp: ${timestamp}

# elapsed time with second resolution
elapsed=$(( end_time - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
