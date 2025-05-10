# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GRPO Train Test Case """

import argparse
from mindrlhf.trainer.spmd.grpo_trainer import GRPOTrainer

def main(input_args):
    trainer = GRPOTrainer(input_args)
    trainer.run_grpo_train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="qwen make experience")
    parser.add_argument("--config", type=str, default=None, help="configs path", required=True)
    parser.add_argument("--sft_path_infer", type=str, default=None, help="sft model path", required=True)
    parser.add_argument("--sft_path_ref", type=str, default=None, help="sft model path", required=True)
    parser.add_argument("--sft_path_train", type=str, default=None, help="sft model path", required=True)
    parser.add_argument("--vocab_path", required=True, help="path to vocab.json")
    parser.add_argument("--merges_file_path", required=True, help="path to merges.txt")
    parser.add_argument("--save_data_file", type=str, default=None, help="save_data_file")
    parser.add_argument("--mind_dataset_dir", type=str, default=None, help="mind_dataset_dir", required=True)
    parser.add_argument("--save_ckpt_dir", type=str, default="./", help="save_ckpt_dir")
    parser.add_argument("--use_parallel", type=str, default=False, help="use_parallel")
    parser.add_argument("--load_sft_checkpoint_infer", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_sft_checkpoint_train", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_ref_checkpoint", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_ckpt_format", type=str, default='ckpt', help="ckpt or safetensors")
    parser.add_argument("--enable_compile_cache", type=str, default=False, help="enable compile cache")
    parser.add_argument("--pre_num_generations", type=int, default=1, help="pre generate times")
    parser.add_argument("--pre_store_data", type=int, default=16, help="pre generate times")
    parser.add_argument("--reward_funcs", nargs='*', type=str, help="reward_funcs")
    parser.add_argument("--reward_weights", nargs='*', type=float, help="reward_weights")
    parser.add_argument("--save_strategy_dir", type=str, default="../../strategy/", help="save_strategy_dir")
    parser.add_argument("--custom_model_name", type=str, default="qwen", help="model name")
    parser.add_argument("--tensorboard_dir", type=str, default="", help="set tensorboard log dir")
    parser.add_argument("--tensorboard_queue_size", type=int, default=10, help="set tensorboard queue size")
    args = parser.parse_args()
    main(args)
