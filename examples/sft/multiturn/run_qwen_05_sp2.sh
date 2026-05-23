#!/bin/bash
set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

# nproc_per_node=1
# save_path='/nobackup2/windy/verl-agent-master/checkpoints/1.5b_webshop_reverkl2e-6'

# # Shift the arguments so $@ refers to the rest
# # shift 2

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
#      -m verl.trainer.posterior_sft_trainer \
#     data.train_files="./data/OpenManus-RL/webshop_train_sft.parquet" \
#     data.val_files="./data/OpenManus-RL/webshop_test_sft.parquet" \
#     data.multiturn.enable=true \
#     data.multiturn.messages_key=messages \
#     data.micro_batch_size=1 \
#     data.max_length=4608 \
#     data.truncation='left' \
#     model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
#     model.use_liger=True \
#     model.enable_gradient_checkpointing=True \
#     optim.lr=2e-6 \
#     trainer.total_epochs=10 \
#     trainer.default_local_dir=$save_path \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='webshop_sft' \
#     trainer.experiment_name='1.5b_reversekl_prior0.2_2e-6' \
#     trainer.default_hdfs_dir=null $@

nproc_per_node=2
save_path='/nobackup2/windy/verl-agent-master/checkpoints/llama3.23b_webshop_sft5e-6'

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/OpenManus-RL/webshop_train_sft.parquet \
    data.val_files=./data/OpenManus-RL/webshop_test_sft.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=32 \
    data.max_length=4608 \
    data.truncation='left' \
    model.partial_pretrain=meta-llama/Llama-3.2-3B-Instruct \
    model.use_liger=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=5e-6 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$save_path \
    trainer.logger=['console','wandb'] \
    trainer.project_name='webshop_sft' \
    trainer.experiment_name='llama3.23b_sft5e-6' \
    trainer.default_hdfs_dir=null 



