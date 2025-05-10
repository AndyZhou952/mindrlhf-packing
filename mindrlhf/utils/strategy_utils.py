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
"""Strategy Utils"""

import os
import copy
import stat
from mindspore import nn
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy
from mindspore.communication.management import get_group_size
from mindformers.experimental.infer.core import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mindformers.tools import logger

from mindrlhf.utils.reshard_optimizer import Layout as StrategyLayout


def _update_sharded_state_dict(network: nn.Cell, dict_: dict):
    """update sharded state dict"""
    cells = network.name_cells()
    for _, subcell in cells.items():
        if subcell == network:
            continue
        if isinstance(subcell, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            dict_.update(subcell.sharded_state_dict())
        else:
            _update_sharded_state_dict(subcell, dict_)


def generate_state_dict(network):
    """generate state dict"""
    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0,
    }
    model_state_dict = {}
    _update_sharded_state_dict(network=network, dict_=model_state_dict)
    # add no shard param
    model_param_dict = network.parameters_dict()
    for name in model_param_dict:
        if name not in model_state_dict:
            model_state_dict[name] = {'shape': model_param_dict[name].shape,
                                      'shard': tuple([1] * model_param_dict[name].ndim)}
    state_dict['model'] = model_state_dict
    model_param_dict = network.parameters_dict()
    for name in model_param_dict:
        if name not in model_state_dict:
            model_state_dict[name] = {'shape': model_param_dict[name].shape,
                                      'shard': tuple([1] * model_param_dict[name].ndim)}
    return state_dict

def save_strategy_file(state_dict, reshard_optimizer, strategy_file_name):
    """save_strategy_file"""
    print(f"----------------start save front parallel strategy---------------")

    stra = ckpt_strategy()

    stage_rank_size = state_dict["stage_rank_size"]
    stage = state_dict["stage"]
    model_param = state_dict["model"]
    stra.current_stage = 0
    for name, item in model_param.items():
        if "shard" not in item or "shape" not in item:
            continue
        opt_weight_shard_step = item["opt_weight_shard_step"] if "opt_weight_shard_step" in item.keys() else 0
        opt_weight_shard_size = item["opt_weight_shard_size"] if "opt_weight_shard_size" in item.keys() else 0
        strategy_item = stra.parallel_strategy_item.add()
        strategy_item.node_name = name
        parallel_strategys = strategy_item.parallel_strategys
        parallel_strategys.stage = stage
        shard = item["shard"]
        shape = item["shape"]
        parallel_strategy = parallel_strategys.parallel_strategy.add()
        shard_mul = 1
        for ele in shard:
            parallel_strategy.dim.append(ele)
            shard_mul = shard_mul * ele
        layout_item = stra.parallel_layout_item.add()
        layout_item.param_name = name
        parallel_layouts = layout_item.parallel_layouts
        parallel_layouts.field = 0
        parallel_layouts.opt_weight_shard_step = opt_weight_shard_step
        parallel_layouts.opt_weight_shard_size = opt_weight_shard_size
        dev_matrix = parallel_layouts.dev_matrix.add()
        repeat_calc_num = 1
        if stage_rank_size == shard_mul:
            repeat_calc_num = 1
        elif stage_rank_size % shard_mul == 0:
            repeat_calc_num = stage_rank_size // shard_mul
        else:
            raise ValueError(
                f"For {name}, the shard{shard} requires {shard_mul} devices, "
                f"but the device number of this stage is {stage_rank_size}, "
                f"it can not be divisible by {shard_mul}"
            )
        if repeat_calc_num != 1:
            dev_matrix.dim.append(repeat_calc_num)
        for ele in shard:
            dev_matrix.dim.append(ele)
        tensor_map = parallel_layouts.tensor_map.add()
        shape_len = len(shape)
        index = shape_len - 1
        for _ in range(shape_len):
            tensor_map.dim.append(index)
            index = index - 1

        if reshard_optimizer:
            src_layout = StrategyLayout(dev_mat=dev_matrix.dim, tensor_map=tensor_map.dim)
            src_layout_origin = copy.deepcopy(src_layout)
            dst_layout = reshard_optimizer.get_dst_layout(src_layout)

            dev_matrix.dim.clear()
            tensor_map.dim.clear()
            dev_matrix.dim.extend(dst_layout.dev_mat)
            tensor_map.dim.extend(dst_layout.tensor_map)

            logger.info(f"Reshard Optimizer change {src_layout_origin} to {dst_layout} in strategy file")

        param_split_shape = parallel_layouts.param_split_shape.add()
        for ele in shape:
            param_split_shape.dim.append(ele)

    try:
        if os.path.exists(strategy_file_name):
            os.chmod(strategy_file_name, stat.S_IWUSR)
        if "/" in strategy_file_name:
            real_path = os.path.abspath(strategy_file_name[: strategy_file_name.rfind("/")])
            os.makedirs(real_path, exist_ok=True)
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(strategy_file_name, flags_, 0o750), "wb") as f:
            f.write(stra.SerializeToString())
            os.chmod(strategy_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical(
            f"Failed to save the checkpoint file {strategy_file_name}. Maybe don't have "
            "the permission to write files, or the disk space is insufficient and so on."
        )
        raise e

    print(f"----------------end save front parallel strategy---------------")
