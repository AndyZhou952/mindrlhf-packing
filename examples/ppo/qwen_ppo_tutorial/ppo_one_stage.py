"""
    run ppo one stage
"""
# Copyright 2024 Huawei Technologies Co., Ltd
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

import os
import argparse
import mindspore.common.dtype as mstype
from mindspore import context, ops
from mindspore.communication.management import get_rank

from mindformers import MindFormerConfig
from mindformers import LlamaConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from mindrlhf.trainer.ppo_trainer import PPOTrainer
from mindrlhf.configs.ppo_configs import PPOConfig
from mindrlhf.utils.configs import (
    combine_config,
    init_ppo_dataset,
    init_network_and_optimizer,
)
from mindrlhf.utils import transfer_from_str_to_bool


def main(sft_path_infer, sft_path_train, reward_path, critic_path, use_parallel, args):
    """
    ppo one stage
    """
    use_parallel = transfer_from_str_to_bool(use_parallel)
    enable_compile_cache = transfer_from_str_to_bool(args.enable_compile_cache)
    only_save_strategy = transfer_from_str_to_bool(args.only_save_strategy)

    # init config with yaml
    sft_config_infer = MindFormerConfig(sft_path_infer)
    sft_config_infer.use_parallel = use_parallel
    os.environ["RUN_MODE"] = sft_config_infer.run_mode

    # init context
    build_context(sft_config_infer)
    build_parallel_config(sft_config_infer)
    context.set_context(
        enable_compile_cache=enable_compile_cache,
        compile_cache_path="./generate_cache",
    )

    # init sft infer model
    sft_config_infer.model.model_config.parallel_config = (
        sft_config_infer.parallel_config
    )
    sft_model_config_infer = LlamaConfig(**sft_config_infer.model.model_config)
    sft_model_config_infer.checkpoint_name_or_path = args.load_sft_checkpoint_infer
    sft_model_config_infer.model_name = "llama"

    # init sft train config
    sft_config_train = MindFormerConfig(sft_path_train)
    sft_config_train.use_parallel = use_parallel
    sft_config_train.model.model_config.parallel_config = (
        sft_config_train.parallel_config
    )
    sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
    sft_model_config_train.checkpoint_name_or_path = args.load_sft_checkpoint_train
    sft_model_config_train.model_name = "llama"

    # init ppo config
    ppo_config = PPOConfig()
    ppo_config.mind_dataset_dir = args.mind_dataset_dir
    ppo_config.save_data_file = args.save_data_file
    ppo_config.only_save_strategy = only_save_strategy
    ppo_config.save_ckpt_dir = args.save_ckpt_dir
    ppo_config.align_type = "rlhf_stages"
    ppo_config.use_parallel = use_parallel
    # ppo_config infer 和 train 共用
    ppo_config = combine_config(ppo_config, sft_model_config_infer)

    # init ref model
    ref_config = MindFormerConfig(sft_path_infer)
    ref_config.use_parallel = use_parallel
    ref_config.model.model_config.parallel_config = ref_config.parallel_config
    ref_model_config = LlamaConfig(**ref_config.model.model_config)
    ref_model_config.checkpoint_name_or_path = args.load_ref_checkpoint
    ref_model_config.model_name = "llama"

    # init reward model
    rm_config = MindFormerConfig(reward_path)
    rm_config.use_parallel = use_parallel
    rm_config.model.model_config.parallel_config = rm_config.parallel_config
    rm_model_config = LlamaConfig(**rm_config.model.model_config)
    rm_model_config.checkpoint_name_or_path = args.load_rm_checkpoint
    rm_model_config.model_name = "llama"

    # init critic model
    critic_config = MindFormerConfig(critic_path)
    critic_config.use_parallel = use_parallel
    critic_config.model.model_config.parallel_config = critic_config.parallel_config
    critic_model_config = LlamaConfig(**critic_config.model.model_config)
    critic_model_config.checkpoint_name_or_path = args.load_critic_checkpoint
    critic_model_config.model_name = "llama"

    trainer = PPOTrainer(
        ppo_config=ppo_config,
        sft_model_config_infer=sft_model_config_infer,
        sft_model_config_train=sft_model_config_train,
        ref_model_config=ref_model_config,
        rm_model_config=rm_model_config,
        critic_model_config=critic_model_config,
    )
    fake_data = {
        "advantages": ops.zeros((2, 512), mstype.float32),
        "attention_mask": ops.zeros((2, 8192), mstype.int32),
        "logprobs": ops.zeros((2, 512), mstype.float32),
        "loss_mask": ops.zeros((2, 8192), mstype.int32),
        "pretrain_ids": ops.zeros((2, 8193), mstype.int32),
        "query_tensors": ops.zeros((2, 4096), mstype.int32),
        "response_tensors": ops.zeros((2, 8192), mstype.int32),
        "returns": ops.zeros((2, 512), mstype.float32),
        "rewards": ops.zeros((2, 512), mstype.float32),
        "values": ops.zeros((2, 512), mstype.float32),
    }
    # 构造输入数据，编译模型，得到分布式策略
    trainer.pre_run(input_data=fake_data)

    # ================= pre compile model for loading dist ckpt ========================
    ppo_with_grad = init_network_and_optimizer(trainer)
    ppo_with_grad.set_train(True)
    ppo_with_grad.compile(**fake_data)

    trainer.ppo_model_infer.compile(**fake_data)
    # trainer.ref_model.compile(fake_data, samples=fake_data)
    # trainer.reward_fn.compile(fake_data)
    # ====================================================================================
    trainer.load_checkpoint()

    for _ in range(ppo_config.epochs):
        trainer.make_experience(
            num_rollouts=ppo_config.num_rollouts, rank_id=get_rank()
        )

        dataset = init_ppo_dataset(trainer)
        trainer.train(ppo_with_grad, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qwen make experience")
    parser.add_argument(
        "--sft_path_infer", type=str, default=None, help="sft model path", required=True
    )
    parser.add_argument(
        "--sft_path_train", type=str, default=None, help="sft model path", required=True
    )
    parser.add_argument(
        "--reward_path", type=str, default=None, help="reward model path", required=True
    )
    parser.add_argument(
        "--critic_path", type=str, default=None, help="critic model path", required=True
    )
    parser.add_argument(
        "--save_data_file", type=str, default=None, help="save_data_file"
    )
    parser.add_argument(
        "--mind_dataset_dir",
        type=str,
        default=None,
        help="mind_dataset_dir",
        required=True,
    )
    parser.add_argument("--save_ckpt_dir", type=str,
                        default="./", help="save_ckpt_dir")
    parser.add_argument("--use_parallel", type=str,
                        default=False, help="use_parallel")
    parser.add_argument(
        "--load_sft_checkpoint_infer",
        type=str,
        default=None,
        help="load checkpoint path",
    )
    parser.add_argument(
        "--load_sft_checkpoint_train",
        type=str,
        default=None,
        help="load checkpoint path",
    )
    parser.add_argument(
        "--load_rm_checkpoint", type=str, default=None, help="load checkpoint path"
    )
    parser.add_argument(
        "--load_critic_checkpoint", type=str, default=None, help="load checkpoint path"
    )
    parser.add_argument(
        "--load_ref_checkpoint", type=str, default=None, help="load checkpoint path"
    )
    parser.add_argument(
        "--enable_compile_cache", type=str, default=False, help="enable compile cache"
    )
    parser.add_argument(
        "--only_save_strategy", type=str, default=False, help="only save strategy"
    )
    my_args = parser.parse_args()
    main(
        my_args.sft_path_infer,
        my_args.sft_path_train,
        my_args.reward_path,
        my_args.critic_path,
        my_args.use_parallel,
        my_args
    )
