#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2
export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign,valid,graph_build"
export MS_DEV_RUNTIME_CONF="memory_statistics:False"
export MS_ENABLE_LCCL=on  # turn off on A3
export HCCL_IF_BASEE_PORT=60009

root_path="$(realpath "$(dirname "$0")")"
root_path=$root_path/../../../
cd $root_path
WORKDIR=$PWD/../
export PYTHONPATH=$root_path:$PYTHONPATH  # define mindrlhf path
export MINDFORMERS_PATH=$WORKDIR/mindformers/
export PYTHONPATH=$MINDFORMERS_PATH:$PYTHONPATH  # define mindformers path

noderank=$1
master_addr=$2
echo $noderank
msrun --worker_num=32 --local_worker_num=8 --node_rank=$noderank --master_addr=$master_addr \
--master_port=9190 --join=False --log_dir=./qwen2_one_log_$noderank \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--sft_path_infer /path/to/predict_xxx.yaml \
--sft_path_train /path/to/finetune_xxx.yaml \
--vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--mind_dataset_dir /path/to/grpo_data/cvalues_one.mindrecord \
--save_data_file ./grpo_1024_$noderank.mindrecord \
--save_ckpt_dir /path/to/ckpt/train \
--use_parallel True \
--load_sft_checkpoint_infer "/path/to/ckpt/infer" \
--load_sft_checkpoint_train "/path/to/ckpt/train/" \
--load_ref_checkpoint "/path/to/ckpt/infer" \
--enable_compile_cache False \
