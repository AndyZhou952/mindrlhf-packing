#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

export MS_ENABLE_LCCL=off
export GLOG_v=3

WORKDIR="$(realpath "$(dirname "$0")")"
echo "WORKDIR is $WORKDIR"
cd $WORKDIR
export MINDRLHF_PATH=$WORKDIR/../../
export MINDFORMERS_PATH=$WORKDIR/mindformers/
export PYTHONPATH=$MINDRLHF_PATH:$MINDFORMERS_PATH:$PYTHONPATH
echo "PYTHONPATH is $PYTHONPATH"

jsonl_path="$WORKDIR/qwen2_5/mini_gsm8k.jsonl"
vocab_path="$WORKDIR/qwen2_5/vocab.json"
merges_path="$WORKDIR/qwen2_5/merges.txt"
mkdir -p $WORKDIR/dataset/
data_path="$WORKDIR/dataset/mini_gsm8k.mindrecord"

python ./qwen2_5/rlhf_data.py \
--vocab_path $vocab_path \
--merges_file_path $merges_path \
--file_path $jsonl_path \
--output_path $data_path > $WORKDIR/data_preprocess.log

mkdir -p $WORKDIR/grpo_data

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
--master_port=9190 --join=True --log_dir=$WORKDIR/qwen2_one_log \
./qwen2_5/grpo_train.py \
--config ./qwen2_5/grpo_config_st.yaml \
--sft_path_infer ./qwen2_5/predict_qwen2_5_7b_instruct_st.yaml \
--sft_path_ref ./qwen2_5/ref_qwen2_5_7b_instruct_st.yaml \
--sft_path_train ./qwen2_5/finetune_qwen2_5_7b_st.yaml \
--vocab_path $vocab_path \
--merges_file_path $merges_path \
--mind_dataset_dir $data_path \
--save_data_file $WORKDIR/grpo_data/grpo.mindrecord \
--save_ckpt_dir $WORKDIR/ckpt/train \
--use_parallel True \
--load_sft_checkpoint_infer "" \
--load_sft_checkpoint_train "" \
--load_ref_checkpoint "" \
--enable_compile_cache False \
--reward_funcs "format_reward" \
--reward_weights 1.0
