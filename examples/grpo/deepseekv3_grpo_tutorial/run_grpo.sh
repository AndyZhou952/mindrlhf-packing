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

export GLOG_v=2
export PYTHONPATH=/path/to/mindrlhf/:$PYTHONPATH
export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH
# use for vllm
export vLLM_MODEL_BACKEND=MindFormers
export MINDFORMERS_MODEL_CONFIG="./model_configs/deepseek_v3_grpo/predict_deepseek3_671b.yaml"
# 性能优化
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_TOTAL_MEMORY_GB=64
export vLLM_MODEL_MEMORY_USE_GB=50
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
export MS_DEV_RUNTIME_CONF="multi_stream:true"
#ray组网相关
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
#通信下发
export HCCL_OP_EXPANSION_MODE=AIV
# 关闭多机lccl
export MS_ENABLE_LCCL=off

master_ip=127.0.0.1
node_rank=$1

bash scripts/msrun_launcher.sh "examples/grpo/deepseekv3_grpo_tutorial/main.py \
--config examples/grpo/deepseekv3_grpo_tutorial/grpo_config.yaml \
--sft_path_infer ./model_configs/deepseek_v3_config/predict_deepseek3_671b.yaml \
--sft_path_ref ./model_configs/deepseek_v3_config/ref_deepseek3_671b.yaml \
--sft_path_train ./model_configs/deepseek_v3_config/finetune_deepseek3_671b.yaml \
--tokenizer_path ./tokenizer.json \
--mind_dataset_dir ./output.mindrecord \
--use_parallel True \
--load_sft_checkpoint_infer /path/to/ckpt/infer \
--load_sft_checkpoint_train /path/to/ckpt/train/ \
--load_ref_checkpoint /path/to/ckpt/infer \
--load_ckpt_format ckpt \
--save_data_file ./grpo_1024.mindrecord \
--save_ckpt_dir /path/to/ckpt/train \
--enable_compile_cache False" \
16 16 $master_ip 8118 $node_rank output/msrun_log False 7200