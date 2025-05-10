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
"""Resharding Weight"""
import numpy as np

import mindspore.log as logger
from mindspore import ops, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.api import _pynative_executor
from mindspore.communication import get_rank, get_group_size
from mindspore.parallel.shard import Layout, _DistributedTensorInfo
from mindspore.parallel._parallel_serialization import _build_searched_strategy, _convert_to_list
from mindspore.parallel.function.reshard_func import _redistribute


def _get_used_dev_mat(dev_mat, tensormap):
    used_dev_mat = []
    for i in range(len(dev_mat)):
        idx = len(dev_mat) - i - 1
        used_dev_mat.append(idx in tensormap)
    return used_dev_mat


def _tensor_map_flatten(tensor_map):
    tensor_map = np.array(tensor_map)
    if tensor_map.ndim == 1:
        return tensor_map.tolist()
    if tensor_map.ndim == 2 and tensor_map.shape[1] == 1:
        return tensor_map.flatten().tolist()
    raise ValueError(f"tensor_map shape: {tensor_map.shape} is not supported")


# pylint: disable=R1705
# pylint: disable=W0212
def _get_dev_mat_for_opt_shard(opt_shard, dev_mat, tensor_map):
    """generate device matrix for opt shard scenario"""
    if opt_shard == 0:
        return dev_mat, -1, tensor_map, True
    used_dev_num = int(np.prod(np.array([dev_mat[len(dev_mat) - i - 1] if i != -1 else 1 for i in tensor_map])))
    total_dev_num = int(np.prod(np.array(dev_mat)))
    if opt_shard == -1 or used_dev_num * opt_shard == total_dev_num:
        # 饱和切分不需要调整dev mat
        return dev_mat, -1, tensor_map, True
    # 非饱和根据场景调整dev mat
    remain_dev_num = total_dev_num // (used_dev_num * opt_shard)
    real_remain_dev_num = 1
    used_dev_mat_mask = _get_used_dev_mat(dev_mat, tensor_map)
    counter = -1
    for i, value in enumerate(dev_mat):
        if used_dev_mat_mask[i]:
            # 当前维度已经被使用，不被计算在有优化器并行中
            continue
        if real_remain_dev_num == remain_dev_num:
            # 现有dev mat已经满足需求，不需要调整
            # 通过counter值修改tensor map
            return dev_mat, counter, tensor_map, False
        elif real_remain_dev_num < remain_dev_num:
            real_remain_dev_num *= value
        else:
            # 对counter位置的维度进行拆分
            splitted_dev_value = dev_mat[counter]
            new_dev_mat_value_first = remain_dev_num // (real_remain_dev_num // splitted_dev_value)
            new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
            new_dev_mat = dev_mat[:counter] + \
                          [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
            new_tensor_map = [value if value < len(dev_mat) - counter - 1 else value + 1 for value in tensor_map]
            return new_dev_mat, counter, new_tensor_map, False
        counter += 1
    if real_remain_dev_num == remain_dev_num:
        # 现有dev mat已经满足需求，不需要调整
        # 通过counter值修改tensor map
        return dev_mat, counter, tensor_map, False
    else:
        # 对counter位置的维度进行拆分
        splitted_dev_value = dev_mat[counter]
        new_dev_mat_value_first = remain_dev_num // (real_remain_dev_num // splitted_dev_value)
        new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
        new_dev_mat = dev_mat[:counter] + \
                      [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
        new_tensor_map = [value if value < len(dev_mat) - counter - 1 else value + 1 for value in tensor_map]
        return new_dev_mat, counter, new_tensor_map, False


def _get_tensor_map_for_opt_shard(dev_mat, tensor_map, full_opt_shard, counter, alias_name):
    """generate tensor map for opt shard scenario"""
    used_dev_mat = _get_used_dev_mat(dev_mat, tensor_map)
    if full_opt_shard:
        unused_idx = [len(used_dev_mat) - i - 1 for i in range(len(used_dev_mat)) if not used_dev_mat[i]]
    else:
        unused_idx = []
        real_idx = 0
        for i in range(len(used_dev_mat)):
            if used_dev_mat[i]:
                continue
            if real_idx > counter:
                unused_idx.append(len(used_dev_mat) - i - 1)
            if not used_dev_mat[i]:
                real_idx += 1

    opt_shard_dim = unused_idx
    if tensor_map[0] != -1:
        opt_shard_dim = [tensor_map[0]] + opt_shard_dim
    opt_shard_name = tuple([alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in opt_shard_dim])
    og_name = [alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in tensor_map[1:]]
    new_tensor_map = tuple([opt_shard_name] + og_name)
    return new_tensor_map


def _layout_process(stra_layout, pp_stages):
    """return the layout list."""
    rank_list = None
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    opt_shard = stra_layout[5]
    dev_mat = stra_layout[0]
    tensor_map = stra_layout[1]
    tensor_map = _tensor_map_flatten(tensor_map)
    new_dev_mat, counter, new_tensor_map, full_opt_shard = _get_dev_mat_for_opt_shard(opt_shard, dev_mat,
                                                                                      tensor_map)
    alias_name = [alphabet[i] for i in range(len(new_dev_mat))]
    if opt_shard != 0:
        # 开了优化器并行，需要修改new_tensor_map
        new_tensor_map = _get_tensor_map_for_opt_shard(new_dev_mat, new_tensor_map, full_opt_shard, counter,
                                                       alias_name)
    else:
        new_tensor_map = tuple([alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in new_tensor_map])
    all_dev_num = get_group_size()
    local_dev_num = int(np.prod(np.array(new_dev_mat)))
    if all_dev_num != local_dev_num:
        # 其中一个网络开启了pp，导致实际设备数和计算出的设备数不一致
        dev_num_in_stage = get_group_size() // pp_stages
        stage_id = stra_layout[6][0]
        rank_list = list(
            range(dev_num_in_stage * stage_id, dev_num_in_stage * (stage_id + 1)))
    new_dev_mat = tuple(new_dev_mat)
    alias_name = tuple(alias_name)
    layout = Layout(new_dev_mat, alias_name, rank_list)
    final_layout = layout(*new_tensor_map)
    logger.debug("The final layout is ", final_layout.to_dict())
    return final_layout


class TransformParametersD2D:
    """
    Transform parameter from source network's layout to destination network's layout. All the parameters will do
    transformation from device to device.

    Args:
         src_network (Cell): The network provides the source parameters.
         dst_network (Cell): The parameters in destination network will be assigned by source network
            after transformation.
        src_strategy_path (str): The path of source merged strategy end with ckpt. Default: None.
        dst_strategy_path (str): The path of destination merged strategy end with ckpt. Default: None.
        match_func (function): Check whether two input parameters are matched. It takes source param and dest param
            as input and return boolean. Default value is None, which means that the parameters name must equal.
            Default: None.
        offload_src (bool): Whether offload the source parameter after transformation. Default: False.
        load_dst (bool): Whether load the destination parameter before assignment. Default: False.

    Raises:
        TypeError: The output of `match_func` is not a boolean.
        ValueError: `src_network` and `dst_network` does not have matched parameter.
        ValueError: `src_strategy_path` or `dst_strategy_path` is None.
    """

    def __init__(self, src_network, dst_network, src_strategy_path=None, dst_strategy_path=None, match_func=None):
        if src_strategy_path is not None:
            src_strategy = _build_searched_strategy(src_strategy_path)
            src_stra_info = _convert_to_list(src_strategy, get_rank())
        else:
            raise ValueError("The src_strategy_path should not be None.")
        if dst_strategy_path is not None:
            dst_strategy = _build_searched_strategy(dst_strategy_path)
            dst_stra_info = _convert_to_list(dst_strategy, get_rank())
        else:
            raise ValueError("The dst_strategy_path should not be None.")

        # 获取src和dst的pp stage个数
        src_pp_stages = max(stra_info[6][0] for stra_info in src_stra_info.values()) + 1
        dst_pp_stages = max(stra_info[6][0] for stra_info in dst_stra_info.values()) + 1

        # 检查src和dst是否是同总卡数倒换
        self._check_total_dev_num(src_stra_info, src_pp_stages, dst_stra_info, dst_pp_stages)

        # 将存在strategy里面的param给存到dict中，不存在strategy.ckpt中的param不做倒换
        # 不能存在在策略文件中的权重也会做倒换，默认按照全1的切分策略并且默认不开启PP，数据放在stage0
        src_net_param_dict = {}
        for name, param in src_network.parameters_and_names():
            src_net_param_dict[name] = param
            if name not in src_stra_info.keys():
                logger.warning(f"for param {name}, it's not in strategy file, set default strategy.")
                src_stra_info[name] = [[get_group_size(),], [-1] * len(param.shape), [], 0, 0, 0, [0]]

        dst_net_param_dict = {}
        for name, param in dst_network.parameters_and_names():
            dst_net_param_dict[name] = param
            if name not in dst_stra_info.keys():
                logger.warning(f"for param {name}, it's not in strategy file, set default strategy.")
                dst_stra_info[name] = [[get_group_size(),], [-1] * len(param.shape), [], 0, 0, 0, [0]]

        # 获取src和dst的param的交集
        src_param_name_intersection = []
        dst_param_name_intersection = []
        for src_name, src_value in src_net_param_dict.items():
            for dst_name, dst_value in dst_net_param_dict.items():
                # 如果有match_func则通过match_func来判断两个网络中的param是否匹配
                if match_func is not None:
                    match_flag = match_func(src_name, dst_name)
                    if not isinstance(match_flag, bool):
                        raise TypeError("The return value of match_func should be bool.")
                    if match_flag:
                        src_layout = _layout_process(src_stra_info[src_name], src_pp_stages)
                        src_value._dtensor_info = _DistributedTensorInfo(src_layout)
                        dst_layout = _layout_process(dst_stra_info[dst_name], dst_pp_stages)
                        dst_value._dtensor_info = _DistributedTensorInfo(dst_layout)
                        src_param_name_intersection.append(src_value)
                        dst_param_name_intersection.append(dst_value)
                        break
                # 如果match_func为None，则需要src_name和dst_name完全匹配
                else:
                    if src_name == dst_name:
                        src_layout = _layout_process(src_stra_info[src_name], src_pp_stages)
                        src_value._dtensor_info = _DistributedTensorInfo(src_layout)
                        dst_layout = _layout_process(dst_stra_info[dst_name], dst_pp_stages)
                        dst_value._dtensor_info = _DistributedTensorInfo(dst_layout)
                        src_param_name_intersection.append(src_value)
                        dst_param_name_intersection.append(dst_value)
                        break
        self._log_not_in_intersection_param(src_param_name_intersection, src_net_param_dict, "src_network")
        self._log_not_in_intersection_param(dst_param_name_intersection, dst_net_param_dict, "dst_network")
        if not src_param_name_intersection or not dst_param_name_intersection:
            raise ValueError("The intersection of src_network and dst_network is empty.")
        if len(src_param_name_intersection) != len(dst_param_name_intersection):
            raise ValueError("The length of src_param_name_intersection and dst_param_name_intersection is not equal.")

        self._src_param_name_intersection = src_param_name_intersection
        self._dst_param_name_intersection = dst_param_name_intersection

        self.assign = P.Assign()

        self._auto_paralell_context_value_map = {}
        self._pipeline_config = {}

    # pylint: disable=W0702
    def transform(self, input_on_device_flag=(True, True)):
        """transform the parameters from source network layout to dest network layout and assign the parameter to
        dest network"""
        if not isinstance(input_on_device_flag, tuple) or len(input_on_device_flag) != 2:
            raise ValueError(f"The input_on_device_flag must be tuple, and its length must be 2."
                             f"But got {type(input_on_device_flag)}")
        for i, flag in enumerate(input_on_device_flag):
            if not isinstance(flag, bool):
                raise TypeError(f"elements in input_on_device_flag must be bool, but got ")
        mem_opt_level = self._check_input_flag(input_on_device_flag)
        logger.info(f"The input on device flag is {input_on_device_flag}, memory optimize level is {mem_opt_level}")
        src_param_name_intersection = self._preprocess_for_src_param(input_on_device_flag)
        for i, src_param in enumerate(src_param_name_intersection):
            if not isinstance(src_param, (Parameter, Tensor)):
                continue
            if mem_opt_level in [1]:
                src_param._load()
            redist_src_param = _redistribute(src_param, self._dst_param_name_intersection[i]._dtensor_info)
            redist_src_param = ops.cast(redist_src_param, self._dst_param_name_intersection[i].dtype)
            if mem_opt_level in [1]:
                src_param._offload()
            if get_rank() in self._dst_param_name_intersection[i]._dtensor_info.layout.to_dict()["rank_list"]:
                _pynative_executor.sync()
                if mem_opt_level in [1]:
                    self._dst_param_name_intersection[i]._load()
                if self._dst_param_name_intersection[i].shape != redist_src_param.shape:
                    try:
                        src_layout = redist_src_param._dtensor_info.layout.to_dict()
                    except:
                        logger.error("Get src layout has error, src_layout will be set to None")
                        src_layout = None
                    try:
                        dst_layout = self._dst_param_name_intersection[i]._dtensor_info.layout.to_dict()
                    except:
                        logger.error("Get dst layout has error, dst_layout will be set to None")
                        dst_layout = None
                    raise ValueError(f"For param {self._dst_param_name_intersection[i].name}, "
                                     f"src shape must be equal to dst shape, but got "
                                     f"src shape {redist_src_param.shape}, "
                                     f"dst shape {self._dst_param_name_intersection[i].shape}. "
                                     f"src layout is {src_layout}, "
                                     f"dst layout is {dst_layout}")
                self.assign(self._dst_param_name_intersection[i], redist_src_param)
                if mem_opt_level in [1]:
                    # to sync dst value
                    self._dst_param_name_intersection[i].asnumpy()
                    self._dst_param_name_intersection[i]._offload()

    # pylint: disable=W0613
    def _preprocess_for_src_param(self, input_on_device_flag):
        return self._src_param_name_intersection

    def _log_not_in_intersection_param(self, param_intersection, all_param, debug_string):
        """find the param in all_param but not in param_intersection"""
        param_intersection_name = [
            param.name for param in param_intersection]
        for param_name in all_param.keys():
            if param_name not in param_intersection_name:
                logger.warning(f"The param {param_name} of {debug_string} is not in the intersection of src_network "
                               f"and dst_network.")

    def _check_total_dev_num(self, src_stra_info, src_pp_stages, dst_stra_info, dst_pp_stages):
        _, first_src_stra_info = next(iter(src_stra_info.items()))
        src_dev_num = np.prod(np.array(first_src_stra_info[0])) * src_pp_stages
        _, first_dst_stra_info = next(iter(dst_stra_info.items()))
        dst_dev_num = np.prod(np.array(first_dst_stra_info[0])) * dst_pp_stages
        if src_dev_num != dst_dev_num:
            raise ValueError(f"src network device number must equal to dest network device number,"
                             f"but got src {src_dev_num}, dst {dst_dev_num}")

    def _check_input_flag(self, input_on_device_flag):
        if input_on_device_flag[0] and input_on_device_flag[1]:
            return 0
        elif not (input_on_device_flag[0] or input_on_device_flag[1]):
            return 1
        else:
            raise ValueError(f"The input_on_device_flag must be in following case (True, True), "
                             f"(False, False), but got {input_on_device_flag}")


class TransformParametersD2DForDSv3(TransformParametersD2D):
    """
    Transform parameter from source network's layout to destination network's layout. All the parameters will do
    transformation from device to device.
    """

    def __init__(self, src_network, dst_network, transform_args, src_strategy_path=None, dst_strategy_path=None,
                 match_func=None):
        super().__init__(src_network, dst_network, src_strategy_path, dst_strategy_path, match_func)
        if not isinstance(transform_args, dict):
            raise TypeError("transform args must be dict")

        self._n_head = transform_args.get("n_head")
        self._qk_nope_head_dim = transform_args.get("qk_nope_head_dim")
        self._qk_rope_head_dim = transform_args.get("qk_rope_head_dim")
        self.l2q_nope_proj = None
        self.l2q_pe_proj = None
        self.kv2l_k_pe = None
        self.kv2l_latent_kv = None
        if len(self._src_param_name_intersection) != len(self._dst_param_name_intersection):
            raise ValueError("The length of src_param_name_intersection and dst_param_name_intersection is not equal.")

    def _preprocess_for_src_param(self, input_on_device_flag):
        """
        preprocess for train param
        """
        mem_opt_level = self._check_input_flag(input_on_device_flag)
        new_src_param_intersection = []
        dev_mat = Layout((get_group_size(),), ("all_dev",))

        for _, src_param in enumerate(self._src_param_name_intersection):
            tensor_map = ["None"] * len(src_param.shape)
            standalone_layout = dev_mat(*tensor_map)
            standalone_dtensor_info = _DistributedTensorInfo(standalone_layout)
            keywords = [
                "feed_forward.routed_experts.ffn.w1",
                "feed_forward.routed_experts.ffn.w2",
                "feed_forward.routed_experts.ffn.w3"
            ]

            if any(keyword in src_param.name for keyword in keywords):
                redist_param = self._redist_func(mem_opt_level, src_param, standalone_dtensor_info)
                redist_param = F.transpose(redist_param, (0, 2, 1))
                if mem_opt_level in [1]:
                    redist_param._offload()
                redist_param._dtensor_info = standalone_dtensor_info
                new_src_param_intersection.append(redist_param)
                continue
            elif "attention.l2q_nope_proj.weight" in src_param.name:
                src_param_obj = self._redist_func(mem_opt_level, src_param, standalone_dtensor_info)
                self._check_value_is_none("l2q_nope_proj", src_param_obj.asnumpy())
                del src_param_obj
            elif "attention.l2q_pe_proj.weight" in src_param.name:
                src_param_obj = self._redist_func(mem_opt_level, src_param, standalone_dtensor_info)
                self._check_value_is_none("l2q_pe_proj", src_param_obj.asnumpy())
                del src_param_obj
            elif "attention.kv2l_k_pe.weight" in src_param.name:
                src_param_obj = self._redist_func(mem_opt_level, src_param, standalone_dtensor_info)
                self._check_value_is_none("kv2l_k_pe", src_param_obj.asnumpy())
                del src_param_obj
            elif "attention.kv2l_latent_kv.weight" in src_param.name:
                src_param_obj = self._redist_func(mem_opt_level, src_param, standalone_dtensor_info)
                self._check_value_is_none("kv2l_latent_kv", src_param_obj.asnumpy())
                del src_param_obj
            else:
                new_src_param_intersection.append(src_param)
                continue

            if (("attention.l2q_nope_proj.weight" in src_param.name or
                 "attention.l2q_pe_proj.weight" in src_param.name) and
                    self.l2q_nope_proj is not None and self.l2q_pe_proj is not None):
                value_nope = self.l2q_nope_proj.reshape(self._n_head, self._qk_nope_head_dim, -1)
                reshaped_l2q_pe_proj = self.l2q_pe_proj.reshape(
                    self._n_head,
                    2,
                    self._qk_rope_head_dim // 2,
                    -1
                )
                transposed_l2q_pe_proj = reshaped_l2q_pe_proj.transpose(0, 2, 1, 3)
                value_pe = transposed_l2q_pe_proj.reshape(
                    self._n_head,
                    self._qk_rope_head_dim,
                    -1
                )
                value_merged = np.concatenate([value_nope, value_pe], axis=1)
                value_merged = value_merged.reshape(-1, value_merged.shape[-1])
                self.l2q_nope_proj = None
                self.l2q_pe_proj = None
            elif ("attention.kv2l_k_pe.weight" in src_param.name or
                  "attention.kv2l_latent_kv.weight" in src_param.name) and \
                    self.kv2l_k_pe is not None and self.kv2l_latent_kv is not None:
                value_k_pe = self.kv2l_k_pe.reshape(2, self._qk_rope_head_dim // 2, -1)
                value_k_pe = value_k_pe.transpose(1, 0, 2).reshape(-1, value_k_pe.shape[-1])
                value_merged = np.concatenate([self.kv2l_latent_kv, value_k_pe], axis=0)
                value_merged = value_merged.reshape(-1, value_merged.shape[-1])
                self.kv2l_k_pe = None
                self.kv2l_latent_kv = None
            else:
                new_src_param_intersection.append("skip")
                continue
            value_merged = Tensor(value_merged)
            value_merged._dtensor_info = standalone_dtensor_info
            if mem_opt_level in [1]:
                value_merged._offload()
            new_src_param_intersection.append(value_merged)
        self._clean_cache()
        return new_src_param_intersection

    def _clean_cache(self):
        self.l2q_nope_proj = None
        self.l2q_pe_proj = None
        self.kv2l_k_pe = None
        self.kv2l_latent_kv = None

    def _check_value_is_none(self, input_var, value):
        if getattr(self, input_var) is None:
            setattr(self, input_var, value)
        else:
            raise ValueError(f"{input_var} has value")

    def _redist_func(self, mem_opt_level, param, dtensor_info):
        if mem_opt_level in [1]:
            param._load()
        param_obj = _redistribute(param, dtensor_info)
        if mem_opt_level in [1]:
            param._offload()
        return param_obj
