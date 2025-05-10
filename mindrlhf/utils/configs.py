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
"""
MindRLHF config
"""
import copy
import math
from dataclasses import asdict, make_dataclass

from mindformers import AutoConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.register import MindFormerConfig
import mindspore
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset, MindDataset
from mindspore.dataset.transforms import TypeCast
from mindspore.nn import PipelineCell, MicroBatchInterleaved
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindrlhf.configs.ppo_configs import PPOConfig
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.dataset import GRPOIteratorStore
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.wrapper import TrainOneStepWithLossScale, TrainPipelineWithLossScaleCell, TrainOneStepWithLossScaleGRPO, \
    TrainPipelineWithLossScaleCellGRPO

__all__ = ['combine_config', 'init_configs']




def set_weight_decay(params, is_use_other_params=True):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """

    def decay_filter(x):
        return 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()

    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    if is_use_other_params:
        group_params = [{
            'params': decay_params,
            'weight_decay': 1e-1
        }, {
            'params': other_params,
            'weight_decay': 0.0
        }, {
            'order_params': params
        }]
    else:
        # use for deepseek
        group_params = [{
            'params': decay_params,
            'weight_decay': 1e-1
        }, {
            'order_params': params
        }]
    return group_params


def combine_config(ppo_config, model_config):
    config_temp = asdict(ppo_config)
    for k, v in model_config.to_dict().items():
        if k not in config_temp:
            config_temp[k] = v
    config_temp['max_prompt_length'] = config_temp['seq_length'] - config_temp['max_decode_length']
    ppo_config_ = make_dataclass("PPOConfig", [(key, type(value)) for key, value in config_temp.items()])
    return ppo_config_(**config_temp)


def init_configs(args=None):
    """
    init configs
    """
    ppo_config = PPOConfig()
    if args:
        ppo_config.mind_dataset_dir = args.dataset_dir
        ppo_config.sft_model_path = args.sft_model_path
        ppo_config.reward_model_path = args.reward_model_path
        ppo_config.critic_model_path = args.critic_model_path
        ppo_config.save_data_file = args.save_data_file
        ppo_config.align_type = args.align_type
    sft_model_path = ppo_config.sft_model_path
    critic_model_path = ppo_config.critic_model_path
    reward_model_path = ppo_config.reward_model_path

    config = MindFormerConfig(sft_model_path)
    build_parallel_config(config)
    sft_model_config = AutoConfig.from_pretrained(sft_model_path)
    sft_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    sft_model_config.parallel = copy.deepcopy(config.parallel)
    sft_model_config.model_name = config.trainer.model_name
    ppo_config.use_past = config.model.model_config.use_past

    ref_model_config = AutoConfig.from_pretrained(sft_model_path)
    ref_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    ref_model_config.parallel = copy.deepcopy(config.parallel)
    ref_model_config.use_past = False
    ref_model_config.model_name = config.trainer.model_name

    config = MindFormerConfig(critic_model_path)
    build_parallel_config(config)
    critic_model_config = AutoConfig.from_pretrained(critic_model_path)
    critic_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    critic_model_config.parallel = copy.deepcopy(config.parallel)
    critic_model_config.use_past = False
    critic_model_config.model_name = config.trainer.model_name

    config = MindFormerConfig(reward_model_path)
    build_parallel_config(config)
    rm_model_config = AutoConfig.from_pretrained(reward_model_path)
    rm_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    rm_model_config.parallel = copy.deepcopy(config.parallel)
    rm_model_config.use_past = False
    rm_model_config.model_name = config.trainer.model_name

    if ppo_config.use_past:
        sft_model_config.batch_size = ppo_config.chunk_size
        ref_model_config.batch_size = ppo_config.chunk_size
        critic_model_config.batch_size = ppo_config.chunk_size
        rm_model_config.batch_size = ppo_config.chunk_size
    ppo_config.model_name = sft_model_config.model_name
    ppo_config = combine_config(ppo_config, sft_model_config)
    print("[PPO Configure] is: ", ppo_config, flush=True)
    print("[ACT Configure] is: ", sft_model_config, sft_model_config.parallel_config, flush=True)
    print("[REF Configure] is: ", ref_model_config, ref_model_config.parallel_config, flush=True)
    print("[CRT Configure] is: ", critic_model_config, critic_model_config.parallel_config, flush=True)
    print("[RM Configure] is: ", rm_model_config, rm_model_config.parallel_config, flush=True)

    return ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config


def init_network_and_optimizer(trainer):
    '''init network and optimizer'''
    sft_model_config = trainer.sft_model_config_train
    ppo_config = trainer.ppo_config
    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        ppo_with_loss_net = PipelineCell(MicroBatchInterleaved(trainer.ppo_model_train,
                                                               ppo_config.micro_batch_interleaved),
                                         sft_model_config.parallel_config.micro_batch_num)
    else:
        print("non-pipeline cell")
        ppo_with_loss_net = trainer.ppo_model_train
    ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)
    lr = LearningRate(learning_rate=ppo_config.start_lr, end_learning_rate=ppo_config.end_lr,
                      warmup_steps=ppo_config.warmup_step, decay_steps=ppo_config.decay_steps)
    params = ppo_with_loss.trainable_params()
    group_params = set_weight_decay(params)

    if ppo_config.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif ppo_config.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=ppo_config.eps, beta1=ppo_config.beta1,
                                      beta2=ppo_config.beta2, param_init_type=sft_model_config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=ppo_config.beta1,
                                             beta2=ppo_config.beta2, eps=ppo_config.eps)

    loss_scale_value = math.pow(2, 12)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2, scale_window=1000)

    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        ppo_with_grad = TrainPipelineWithLossScaleCell(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                       scale_update_cell=update_cell)
    else:
        print("non-pipeline cell")
        ppo_with_grad = TrainOneStepWithLossScale(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                  scale_update_cell=update_cell, enable_global_norm=True)
    return ppo_with_grad


def init_ppo_dataset(trainer):
    """
    init ppo dataset
    """
    ppo_config = trainer.ppo_config
    sft_model_config = trainer.sft_model_config_train
    column_names = ["prompt_completion_ids", "responses_mask",
                    "ref_per_token_logps", "advantages",
                    "actual_sequence_length", "sample_index", "sample_valid_length"]
    # if ppo_config.save_data_file and 'stages' in ppo_config.align_type:
    if not trainer.store:
        dataset = MindDataset(dataset_files=ppo_config.save_data_file, shuffle=False)
        dataset = dataset.project(columns=column_names)
    else:
        pipeline = IteratorStore(trainer.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
    type_cast_op_int32 = TypeCast(mindspore.int32)
    type_cast_op_fp16 = TypeCast(mindspore.float16)
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="query_tensors")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="response_tensors")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="logprobs")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="values")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="rewards")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="returns")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="pretrain_ids")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="loss_mask")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="attention_mask")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="actual_sequence_length")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_index")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_valid_length")
    micro_batch_num = 1
    if sft_model_config.parallel_config.pipeline_stage > 1:
        micro_batch_num = sft_model_config.parallel_config.micro_batch_num
    dataset = dataset.batch(
        batch_size=ppo_config.batch_size * sft_model_config.parallel_config.data_parallel * micro_batch_num)
    return dataset


def combine_grpo_config(grpo_config, model_config):
    config_temp = asdict(grpo_config)
    for k, v in model_config.to_dict().items():
        if k not in config_temp:
            config_temp[k] = v
    config_temp['max_prompt_length'] = config_temp['seq_length'] - config_temp['max_decode_length']
    grpo_config_ = make_dataclass("GRPOConfig", [(key, type(value)) for key, value in config_temp.items()])
    return grpo_config_(**config_temp)


def init_grpo_dataset(trainer):
    """
    init grpo dataset
    """
    grpo_config = trainer.grpo_config
    sft_model_config = trainer.sft_model_config_train
    column_names = ["prompt_completion_ids", "responses_mask",
                    "ref_per_token_logps", "advantages",
                    "actual_sequence_length", "sample_index", "sample_valid_length"]
    # if grpo_config.save_data_file and 'stages' in grpo_config.align_type:
    if not trainer.store:
        dataset = MindDataset(dataset_files=grpo_config.save_data_file, shuffle=False)
        dataset = dataset.project(columns=column_names)
    else:
        pipeline = GRPOIteratorStore(trainer.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
    type_cast_op_int32 = TypeCast(mindspore.int32)
    type_cast_op_fp16 = TypeCast(mindspore.float16)
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompt_completion_ids")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="responses_mask")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="ref_per_token_logps")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="actual_sequence_length")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_index")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_valid_length")
    micro_batch_num = 1
    if sft_model_config.parallel_config.pipeline_stage > 1:
        micro_batch_num = sft_model_config.parallel_config.micro_batch_num

    print(f"##################### bs:{grpo_config.batch_size}, dp:{sft_model_config.parallel_config.data_parallel}, micro_batch_num:{micro_batch_num}")
    dataset = dataset.batch(
        batch_size=grpo_config.batch_size * sft_model_config.parallel_config.data_parallel * micro_batch_num)
    return dataset

def init_grpo_network_and_optimizer(trainer):
    '''init grpo network and optimizer'''
    sft_model_config = trainer.sft_model_config_train
    grpo_config = trainer.grpo_config
    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        grpo_with_loss_net = PipelineCell(MicroBatchInterleaved(trainer.grpo_model_train,
                                                                grpo_config.micro_batch_interleaved),
                                          sft_model_config.parallel_config.micro_batch_num)
    else:
        print("non-pipeline cell")
        grpo_with_loss_net = trainer.grpo_model_train
    grpo_with_loss = _VirtualDatasetCell(grpo_with_loss_net)
    lr = LearningRate(learning_rate=grpo_config.start_lr, end_learning_rate=grpo_config.end_lr,
                      warmup_steps=grpo_config.warmup_step, decay_steps=grpo_config.decay_steps)
    params = grpo_with_loss.trainable_params()
    if trainer.sft_model_config_train.model_name == "deepseek_training":
        group_params = set_weight_decay(params, is_use_other_params=False)
    else:
        group_params = set_weight_decay(params)

    if grpo_config.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif grpo_config.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=grpo_config.eps, beta1=grpo_config.beta1,
                                      beta2=grpo_config.beta2, param_init_type=sft_model_config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=grpo_config.beta1,
                                             beta2=grpo_config.beta2, eps=grpo_config.eps)

    loss_scale_value = math.pow(2, 12)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2, scale_window=1000)

    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        grpo_with_grad = TrainPipelineWithLossScaleCellGRPO(grpo_with_loss, optimizer=optimizer,
                                                            config=sft_model_config, scale_update_cell=update_cell)
    else:
        print("non-pipeline cell")
        grpo_with_grad = TrainOneStepWithLossScaleGRPO(grpo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                       scale_update_cell=update_cell, enable_global_norm=True)
    return grpo_with_grad
