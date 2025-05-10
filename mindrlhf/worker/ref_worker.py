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
"""Reference Worker"""

# python
import time
import os
from glob import glob
import numpy as np


# mindspore
import mindspore as ms
from mindspore import context, ops
from mindspore.communication.management import get_rank, create_group

# mindformers
from mindformers import MindFormerConfig
from mindformers import logger
from mindformers import LlamaConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from research.deepseek3.deepseek3_config import DeepseekV3Config

# mindrlhf
from mindrlhf.models.grpo_models import CausalLMHybrid
from mindrlhf.worker.worker import Worker
from mindrlhf.utils import print_perf_stat, _get_pipeline_group


class RefWorker(Worker):
    """
    This class generates responses.
    """

    def __init__(self, grpo_config, sft_path_ref, args):
        super().__init__()
        logger.info("init RefWorker")
        self.args = args
        self.use_parallel = grpo_config.use_parallel
        ref_config = MindFormerConfig(sft_path_ref)
        ref_config.use_parallel = args.use_parallel
        ref_config.model.model_config.parallel_config = ref_config.parallel_config
        self.ref_pp_stage = ref_config.parallel_config.pipeline_stage
        self.ref_config = ref_config
        ref_config.model.model_config.use_past = False
        if args.custom_model_name in ["qwen", "llama"]:
            ref_model_config = LlamaConfig(**ref_config.model.model_config)
            ref_model_config.model_name = "llama"
        elif args.custom_model_name == "deepseek":
            ref_config.model.model_config.moe_config = ref_config.moe_config
            ref_model_config = DeepseekV3Config(**ref_config.model.model_config)
            ref_model_config.model_name = "deepseek_training"
        else:
            raise ValueError(
                f"model_name should in ['qwen', 'llama','deepseek'], but get {args.custom_model_name}"
            )

        # set pipeline stage
        context.set_auto_parallel_context(
            parallel_mode="semi_auto_parallel", full_batch=True
        )
        context.set_auto_parallel_context(
            pipeline_stages=self.ref_pp_stage, enable_parallel_optimizer=False
        )
        logger.info(f"ref_model_config:{ref_model_config}")
        # set allreduce
        rank_list, pipeline_group_name = _get_pipeline_group()
        pipeline_group_name = "ref_pipeline" + pipeline_group_name
        logger.info(f"start create pipeline {pipeline_group_name}")
        create_group(pipeline_group_name, rank_list)
        logger.info(f"end create pipeline {pipeline_group_name}")
        self.all_reduce = ops.AllReduce(group=pipeline_group_name)
        ref_model_config.checkpoint_name_or_path = args.load_ref_checkpoint
        self.ref_model_config = ref_model_config
        self.ref_ckpt_path = self.get_ref_ckpt_path
        ref_model_config.checkpoint_name_or_path = None
        self.ref_model = CausalLMHybrid(ref_model_config, grpo_config)
        self.ref_model.model.set_train(False)
        for name, param in self.ref_model.parameters_and_names():
            param.name = name
        self.on_device = True
        self.grpo_config = grpo_config
        self.ref_pp_stage = ref_config.parallel_config.pipeline_stage or 1
        self.ref_dp = ref_config.parallel_config.data_parallel
        self.save_strategy_dir = grpo_config.save_strategy_dir

    def model(self):
        return self.ref_model

    @property
    def get_ref_ckpt_path(self):
        return self.ref_model_config.checkpoint_name_or_path

    def get_ref_dp(self):
        return self.ref_dp

    def compile(self):
        """compile and save strategy"""
        self.ref_model.model.set_train(False)
        context.set_auto_parallel_context(
            pipeline_stages=self.ref_pp_stage, enable_parallel_optimizer=False
        )
        total_ref_model_batch_size = self.grpo_config.ref_model_batch_size * self.ref_dp
        fake_data = ms.Tensor(
            shape=(total_ref_model_batch_size, self.grpo_config.seq_length),
            dtype=ms.int32,
        )
        actual_seq_data = ms.Tensor(
            shape=(total_ref_model_batch_size, self.grpo_config.pack_num),
            dtype=ms.int32,
        )
        stage_name = "infer"
        strategy_path = self.grpo_config.save_strategy_dir
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{strategy_path}/{stage_name}_ref_strategy/strategy_{get_rank()}.ckpt"
            }
        )
        # To avoid mindspore compiler's unpacking bug and prevent duplicate compilation,
        # use positional arguments instead of keyword arguments
        self.ref_model.compile(
            fake_data,
            fake_data,
            samples=fake_data,
            actual_seq_length=actual_seq_data,
            is_ref=False,
        )
        stage_name = "other"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{strategy_path}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"
            }
        )

    def compute_ref_log_prob(
            self,
            prompt_completion_ids_tensor,
            attention_mask_tensor,
            samples,
            actual_sequence_length,
        ):
        """
        compute ref log prob
        """
        np.set_printoptions(threshold=1024)
        context.set_auto_parallel_context(
            pipeline_stages=self.ref_pp_stage, enable_parallel_optimizer=False
        )
        logger.info(
            f"precision refmodel inputs are {prompt_completion_ids_tensor}, {samples}"
        )

        context.set_auto_parallel_context(
            parallel_mode="semi_auto_parallel", full_batch=True
        )
        ref_per_token_logps = self.ref_model(
            prompt_completion_ids_tensor,
            attention_mask_tensor,
            samples=samples,
            actual_seq_length=actual_sequence_length,
        )
        if self.ref_pp_stage > 1:
            ref_per_token_logps = self.all_reduce(ref_per_token_logps)
        return ref_per_token_logps

    def offload(self):
        """offload ref model"""
        if self.on_device is False:
            return
        logger.info(f"before offload ref model {ms.hal.memory_stats()}")
        start_time = time.time()
        for param in self.ref_model.get_parameters(expand=True):
            # pylint: disable=W0212
            param._offload()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload ref model")
        logger.info(f"after offload ref model {ms.hal.memory_stats()}")
        self.on_device = False

    def load(self):
        """load ref model"""
        if self.on_device:
            return
        logger.info(f"before load ref model {ms.hal.memory_stats()}")
        start_time = time.time()
        for param in self.ref_model.get_parameters(expand=True):
            # pylint: disable=W0212
            param._load()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load ref model")
        logger.info(f"after load ref model {ms.hal.memory_stats()}")
        self.on_device = True

    def load_checkpoint(self):
        """load_checkpoint"""
        if self.ref_ckpt_path and self.args.load_ckpt_format == "safetensors":
            self.on_device = True
            self._load_checkpoint_safetensors()
            return
        load_ckpt_func = (
            load_distributed_checkpoint if self.use_parallel else ms.load_checkpoint
        )
        logger.info(
            f"self.grpo_config.use_parallel is {self.use_parallel} {load_ckpt_func}"
        )
        if self.ref_ckpt_path:
            self.on_device = True
            param_dict = load_ckpt_func(self.ref_ckpt_path)
            new_param_dict = {"model." + k: v for k, v in param_dict.items()}
            # ===========================================================================
            logger.info(f"begin to load ref model from: {self.ref_ckpt_path}")
            for _, param in self.ref_model.parameters_and_names():
                logger.info(f"ref model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(
                self.ref_model, new_param_dict
            )
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        network = self.ref_model.model
        prefix = "model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {
            f"{prefix}{key}": value for key, value in weight_dict.items()
        }
        return new_weight_dict

    def _load_checkpoint_safetensors(self):
        """load safetensors checkpoint"""
        network = self.ref_model.model
        prefix = "model."
        name_map = None
        try:
            load_checkpoint_files = glob(
                os.path.join(self.ref_ckpt_path, f"*.safetensors")
            )
            load_checkpoint_files.sort()
            name_map = network.obtain_name_map(load_checkpoint_files)
            name_map = {f"{prefix}{key}": value for key, value in name_map.items()}
        except Exception as e:
            raise TypeError(
                f"Please complete abstract function obtain_name_map. Details: {e}"
            ) from e

        # TODO: save strategy
        strategy_path = os.path.join(
            self.save_strategy_dir, "merge_strategy", "infer_ref_merged_strategy.ckpt"
        )
        ms.load_distributed_checkpoint(
            network=self.ref_model,
            predict_strategy=strategy_path,
            unified_safetensors_dir=self.ref_ckpt_path,
            format="safetensors",
            name_map=name_map,
        )
