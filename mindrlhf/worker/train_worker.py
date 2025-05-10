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
"""Train Worker"""

# python
import os
import time
import math
from glob import glob

# mindspore
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset import GeneratorDataset
from mindspore.communication import get_rank
from mindspore import context
from mindspore.nn.wrap.cell_wrapper import (
    PipelineCell,
    _VirtualDatasetCell,
    MicroBatchInterleaved
)
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

# mindformers
from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers.core.callback.callback import TopkBiasBalanceCallback
from mindformers import LlamaConfig
from mindformers import logger
from research.deepseek3.deepseek3_config import DeepseekV3Config

# mindrlhf
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay, print_perf_stat
from mindrlhf.wrapper import (
    TrainOneStepWithLossScaleGRPO,
    TrainPipelineWithLossScaleCellGRPO,
)
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelTrain
from mindrlhf.utils.dataset import GRPOIteratorStore
from mindrlhf.worker.worker import Worker
from mindrlhf.utils.configs import set_weight_decay


class TrainWorker(Worker):
    """
    This class do grpo train.
    """

    def __init__(self, grpo_config, sft_path_train, args):
        super().__init__()
        logger.info("init TrainWorker")
        self.args = args
        self.grpo_config = grpo_config
        sft_config_train = MindFormerConfig(sft_path_train)
        sft_config_train.use_parallel = args.use_parallel
        self.sft_config_train = sft_config_train
        sft_config_train.model.model_config.parallel_config = (
            sft_config_train.parallel_config
        )
        os.environ["RUN_MODE"] = sft_config_train.run_mode
        sft_config_train.model.model_config.parallel_config.recompute = (
            sft_config_train.recompute_config
        )
        if args.custom_model_name in ["qwen", "llama"]:
            sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
            sft_model_config_train.model_name = "llama"
        elif args.custom_model_name == "deepseek":
            sft_config_train.model.model_config.moe_config = sft_config_train.moe_config
            sft_model_config_train = DeepseekV3Config(
                **sft_config_train.model.model_config
            )
            sft_model_config_train.model_name = "deepseek_training"
            self.topk_bias_balance_callback = TopkBiasBalanceCallback(
                sft_model_config_train.moe_config.balance_via_topk_bias,
                sft_model_config_train.moe_config.topk_bias_update_rate,
                sft_model_config_train.num_layers,
                sft_model_config_train.mtp_depth,
                sft_model_config_train.moe_config.expert_num,
                sft_model_config_train.parallel_config.micro_batch_num,
            )
        else:
            raise ValueError(
                f"model_name should in ['qwen', 'llama','deepseek'], but get {args.custom_model_name}"
            )
        sft_model_config_train.checkpoint_name_or_path = args.load_sft_checkpoint_train
        self.sft_ckpt_path_train = sft_model_config_train.checkpoint_name_or_path
        sft_model_config_train.checkpoint_name_or_path = None

        self.train_pp_stage = sft_model_config_train.parallel_config.pipeline_stage or 1
        self.enable_parallel_optimizer = (
            sft_config_train.parallel.enable_parallel_optimizer
            and sft_model_config_train.parallel_config.data_parallel > 1
        )
        context.set_auto_parallel_context(
            pipeline_stages=self.train_pp_stage,
            enable_parallel_optimizer=self.enable_parallel_optimizer,
        )
        self.sft_model_config_train = sft_model_config_train
        policy_model = CausalLMHybrid(sft_model_config_train, self.grpo_config)
        self.grpo_model_train = GRPOModelTrain(grpo_config, policy_model)
        self.grpo_model_train.set_train(True)
        self.grpo_with_grad = self._init_grpo_network_and_optimizer()
        self.store = []

        self.model_on_device = True
        self.optimizer_on_device = True
        self.save_strategy_dir = grpo_config.save_strategy_dir

        self.tensor_writer = self.args.tensor_writer
        self.global_training_step = 0

    def model(self):
        return self.grpo_model_train

    def compile(self):
        """compile and save strategy"""
        self.grpo_with_grad.set_train(True)
        self.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)
        context.set_auto_parallel_context(
            pipeline_stages=self.train_pp_stage,
            enable_parallel_optimizer=self.enable_parallel_optimizer,
        )
        if self.train_pp_stage == 1:
            # for pipeline stage 1, the micro_batch_num is not used
            train_bs = (
                self.grpo_config.batch_size
                * self.sft_model_config_train.parallel_config.data_parallel
            )
        else:
            train_bs = (
                self.grpo_config.batch_size
                * self.sft_model_config_train.parallel_config.micro_batch_num
                * self.sft_model_config_train.parallel_config.data_parallel
            )
        prompt_completion_ids = ms.Tensor(
            shape=(train_bs, self.grpo_config.seq_length + 1), dtype=ms.int32
        )  # [bs, seq_len+1]
        responses_mask = ms.Tensor(
            shape=(train_bs, self.grpo_config.seq_length), dtype=ms.int32
        )  # [bs, seq_len]
        ref_per_token_logps = ms.Tensor(
            shape=(train_bs, self.grpo_config.seq_length), dtype=ms.float16
        )  # [bs, seq_len]
        advantages = ms.Tensor(
            shape=(train_bs, self.grpo_config.seq_length), dtype=ms.float16
        )  # [bs, seq_len]
        actual_seq_length = ms.Tensor(
            shape=(train_bs, self.grpo_config.pack_num), dtype=ms.int32
        )  # [bs, packed_sample_num]
        sample_index = ms.Tensor(
            shape=(train_bs, self.grpo_config.seq_length), dtype=ms.int32
        )  # [bs, seq_len]
        sample_valid_len = ms.Tensor(
            shape=(train_bs, self.grpo_config.pack_num), dtype=ms.int32
        )  # [bs, packed_sample_num]

        start_time = time.time()
        stage_name = "train"
        strategy_path = self.grpo_config.save_strategy_dir
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{strategy_path}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"
            },
            pipeline_stages=self.train_pp_stage,
        )
        # To avoid mindspore compiler's unpacking bug and prevent duplicate compilation,
        # use positional arguments instead of keyword arguments
        self.grpo_with_grad.compile(
            prompt_completion_ids,
            responses_mask,
            ref_per_token_logps,
            advantages,
            actual_seq_length,
            sample_index,
            sample_valid_len,
        )
        stage_name = "other"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"
            }
        )
        end_time = time.time()
        print_perf_stat(start_time, end_time, "train model compile")

        sim_level = os.getenv("MS_SIMULATION_LEVEL")
        if sim_level:
            logger.info(f"start dryrun with sim_level: {sim_level}")
            self.grpo_with_grad(
                prompt_completion_ids=prompt_completion_ids,
                responses_mask=responses_mask,
                ref_per_token_logps=ref_per_token_logps,
                advantages=advantages,
                actual_seq_length=actual_seq_length,
                sample_index=sample_index,
                sample_valid_len=sample_valid_len,
            )
            logger.info(f"dryrun finished")
            exit(0)

    def load_checkpoint(self):
        """load_checkpoint"""
        if self.sft_ckpt_path_train and self.args.load_ckpt_format == "safetensors":
            self.on_device = True
            self._load_checkpoint_safetensors()
            return
        load_ckpt_func = (
            load_distributed_checkpoint
            if self.grpo_config.use_parallel
            else ms.load_checkpoint
        )
        logger.info(
            f"self.grpo_config.use_parallel is {self.grpo_config.use_parallel}, {load_ckpt_func}"
        )
        if self.sft_ckpt_path_train:
            self.model_on_device = True
            self.optimizer_on_device = True
            param_dict = load_ckpt_func(self.sft_ckpt_path_train)
            new_param_dict = {
                "grpo_model_train.policy_model.model." + k: v
                for k, v in param_dict.items()
            }
            logger.info(
                f"begin to load train policy model from: {self.sft_ckpt_path_train}"
            )
            for _, param in self.grpo_model_train.grpo_model_train.policy_model.parameters_and_names():
                logger.info(f"train model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(
                self.grpo_model_train.grpo_model_train.policy_model, new_param_dict
            )
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def _init_grpo_network_and_optimizer(self):
        """
        Build train network.
        """

        sft_model_config = self.sft_model_config_train
        grpo_model_train = self.grpo_model_train
        grpo_config = self.grpo_config
        if sft_model_config.parallel_config.pipeline_stage > 1:
            logger.info("pipeline cell")
            grpo_with_loss_net = PipelineCell(MicroBatchInterleaved(grpo_model_train,
                                                                    grpo_config.micro_batch_interleaved),
                                              sft_model_config.parallel_config.micro_batch_num)
        else:
            logger.info("non-pipeline cell")
            grpo_with_loss_net = grpo_model_train
        grpo_with_loss = _VirtualDatasetCell(grpo_with_loss_net)
        lr = LearningRate(
            learning_rate=grpo_config.start_lr,
            end_learning_rate=grpo_config.end_lr,
            warmup_steps=grpo_config.warmup_step,
            decay_steps=grpo_config.decay_steps,
        )
        params = grpo_with_loss.trainable_params()
        if self.args.custom_model_name == "deepseek":
            group_params = set_weight_decay(params, is_use_other_params=False)
        else:
            group_params = set_weight_decay(params)

        if grpo_config.optimizer == "lamb":
            optimizer = nn.Lamb(group_params, learning_rate=lr)
        elif grpo_config.opt_offload:
            optimizer = AdamWeightDecayOp(
                group_params,
                learning_rate=lr,
                eps=grpo_config.eps,
                beta1=grpo_config.beta1,
                beta2=grpo_config.beta2,
                param_init_type=sft_model_config.param_init_type,
            )
        else:
            optimizer = FP32StateAdamWeightDecay(
                group_params,
                learning_rate=lr,
                beta1=grpo_config.beta1,
                beta2=grpo_config.beta2,
                eps=grpo_config.eps,
            )

        loss_scale_value = math.pow(2, 12)
        update_cell = DynamicLossScaleUpdateCell(
            loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000
        )

        if sft_model_config.parallel_config.pipeline_stage > 1:
            logger.info("pipeline cell")
            grpo_with_grad = TrainPipelineWithLossScaleCellGRPO(
                grpo_with_loss,
                optimizer=optimizer,
                config=sft_model_config,
                scale_update_cell=update_cell,
            )
        else:
            logger.info("non-pipeline cell")
            grpo_with_grad = TrainOneStepWithLossScaleGRPO(
                grpo_with_loss,
                optimizer=optimizer,
                config=sft_model_config,
                scale_update_cell=update_cell,
                enable_global_norm=True,
            )
        return grpo_with_grad

    def _init_grpo_dataset_before_train(self):
        """
        Build dataset for graph pre-compilation.
        """
        grpo_config = self.grpo_config
        column_names = [
            "prompt_completion_ids",
            "responses_mask",
            "ref_per_token_logps",
            "advantages",
            "actual_sequence_length",
            "sample_index",
            "sample_valid_length",
        ]
        logger.info(f"store.length: {len(self.store)}")
        pipeline = GRPOIteratorStore(self.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
        logger.info(f"dataset.len: {len(dataset)}")
        type_cast_op_int32 = TypeCast(ms.int32)
        type_cast_op_fp16 = TypeCast(ms.float16)
        dataset = dataset.map(
            operations=type_cast_op_int32, input_columns="prompt_completion_ids"
        )
        dataset = dataset.map(
            operations=type_cast_op_int32, input_columns="responses_mask"
        )
        dataset = dataset.map(
            operations=type_cast_op_fp16, input_columns="ref_per_token_logps"
        )
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
        dataset = dataset.map(
            operations=type_cast_op_int32, input_columns="actual_sequence_length"
        )
        dataset = dataset.map(
            operations=type_cast_op_int32, input_columns="sample_index"
        )
        dataset = dataset.map(
            operations=type_cast_op_int32, input_columns="sample_valid_length"
        )
        micro_batch_num = 1
        if self.sft_model_config_train.parallel_config.pipeline_stage > 1:
            micro_batch_num = (
                self.sft_model_config_train.parallel_config.micro_batch_num
            )
        batch_size = (
            grpo_config.batch_size
            * self.sft_model_config_train.parallel_config.data_parallel
            * micro_batch_num
        )
        logger.info(
            f"bs:{grpo_config.batch_size}, "
            f"dp:{self.sft_model_config_train.parallel_config.data_parallel}, "
            f"micro_batch_num:{micro_batch_num}, "
            f"bs in dataset: {batch_size}"
        )
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        return dataset

    def train(self):
        """train"""
        dataset = self._init_grpo_dataset_before_train()
        context.set_auto_parallel_context(
            pipeline_stages=self.train_pp_stage,
            enable_parallel_optimizer=self.enable_parallel_optimizer,
        )

        def formatter(out):
            return out.asnumpy() if isinstance(out, Tensor) else out

        iterator = dataset.create_dict_iterator()
        logger.info(f"dataset size is {len(dataset)}")
        train_start_time = time.time()
        for step, databatch in enumerate(iterator):

            ep_begin_time = time.time()
            out = self.grpo_with_grad(**databatch)
            end_time = time.time()
            print_perf_stat(ep_begin_time, end_time, f"train step {step}")
            logger.info(
                " loss: {} | lr: {} | is overflow: {} | loss scale: {}".format(
                    formatter(out[0]),
                    formatter(out[1]),
                    formatter(out[2]),
                    formatter(out[3]),
                )
            )
            if self.args.custom_model_name == "deepseek":
                if self.topk_bias_balance_callback.update_topk_bias_flag:
                    policy_model = (
                        self.grpo_model_train.grpo_model_train.policy_model.model
                    )
                    # pylint: disable=W0212
                    self.topk_bias_balance_callback._update_topk_bias(policy_model)
            if self.tensor_writer:
                self.tensor_writer.add_scalar(
                    "loss", out[0].asnumpy(), global_step=self.global_training_step
                )
                self.tensor_writer.add_scalar(
                    "lr", out[1].asnumpy(), global_step=self.global_training_step
                )
                self.tensor_writer.add_scalar(
                    "overflow", out[2].asnumpy(), global_step=self.global_training_step
                )
                self.tensor_writer.add_scalar(
                    "loss-scale",
                    out[3].asnumpy(),
                    global_step=self.global_training_step,
                )
            self.global_training_step += 1
        train_end_time = time.time()
        print_perf_stat(
            train_start_time,
            train_end_time,
            f"train model all steps {dataset.dataset_size}"
        )

    def offload_optimizer(self):
        """offload optimizer"""
        if self.optimizer_on_device is False:
            return
        logger.info(f"before offload train {ms.hal.memory_stats()}")
        start_time = time.time()
        for param in self.grpo_with_grad.optimizer.moments1:
            # pylint: disable=W0212
            param._offload()
        for param in self.grpo_with_grad.optimizer.moments2:
            # pylint: disable=W0212
            param._offload()
        if self.train_pp_stage > 1:
            for param in self.grpo_with_grad.accu_grads:
                # pylint: disable=W0212
                param._offload()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload train optimizer")
        logger.info(f"after offload train {ms.hal.memory_stats()}")
        self.optimizer_on_device = False

    def load_optimizer(self):
        """load optimizer"""
        if self.optimizer_on_device:
            return
        logger.info(f"before load train optimizer {ms.hal.memory_stats()}")
        start_time = time.time()
        for param in self.grpo_with_grad.optimizer.moments1:
            # pylint: disable=W0212
            param._load()
        for param in self.grpo_with_grad.optimizer.moments2:
            # pylint: disable=W0212
            param._load()
        if self.train_pp_stage > 1:
            for param in self.grpo_with_grad.accu_grads:
                # pylint: disable=W0212
                param._load()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load train optimizer")
        logger.info(f"after load train optimizer {ms.hal.memory_stats()}")
        self.optimizer_on_device = True

    def load_model(self):
        """load model"""
        if self.model_on_device:
            return
        start_time = time.time()
        logger.info(f"before load train model {ms.hal.memory_stats()}")
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            # pylint: disable=W0212
            param._load()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load train model")
        logger.info(f"after load train model {ms.hal.memory_stats()}")
        self.model_on_device = True

    def offload_model(self):
        """offload model"""
        if self.model_on_device is False:
            return
        start_time = time.time()
        logger.info(f"after offload train model {ms.hal.memory_stats()}")
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            # pylint: disable=W0212
            param._offload()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload train model")
        logger.info(f"after offload train model {ms.hal.memory_stats()}")
        self.model_on_device = False

    def push_to_store(self, data):
        self.store = data

    def save_checkpoint(self, rank_id=0, steps=0):
        """save checkpoint"""
        if self.grpo_config.save_ckpt_dir:
            logger.info("Save checkpoints in {}".format(self.grpo_config.save_ckpt_dir))
            train_save_dir = os.path.join(
                self.grpo_config.save_ckpt_dir, "train", f"rank_{rank_id}"
            )
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            grpo_filename = os.path.join(
                train_save_dir, "policy_model_epoch_{}.ckpt".format(steps)
            )
            ms.save_checkpoint(
                self.grpo_model_train.grpo_model_train.policy_model,
                grpo_filename,
                integrated_save=False,
            )
        else:
            logger.info("There is no checkpoint to save!")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        network = self.grpo_model_train.grpo_model_train.policy_model.model
        prefix = "grpo_model_train.policy_model.model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {
            f"{prefix}{key}": value for key, value in weight_dict.items()
        }
        return new_weight_dict

    def _load_checkpoint_safetensors(self):
        """load safetensors checkpoint"""
        network = self.grpo_model_train.grpo_model_train.policy_model.model
        prefix = "grpo_model_train.policy_model.model."
        name_map = None
        try:
            load_checkpoint_files = glob(
                os.path.join(self.sft_ckpt_path_train, f"*.safetensors")
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
            self.save_strategy_dir,
            "merge_strategy",
            "train_policy_merged_strategy.ckpt",
        )
        ms.load_distributed_checkpoint(
            network=self.grpo_model_train.grpo_model_train.policy_model,
            predict_strategy=strategy_path,
            unified_safetensors_dir=self.sft_ckpt_path_train,
            format="safetensors",
            name_map=name_map,
        )
