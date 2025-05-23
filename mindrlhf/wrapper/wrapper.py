# Copyright 2021 Huawei Technologies Co., Ltd
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
"""GPT training wrapper"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore import context, Parameter
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from mindrlhf.utils.utils import GlobalNorm, ClipByGlobalNorm

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    # 0 for clip_by_value and 1 for clip_by_norm
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
            F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad,
                                   F.cast(F.tuple_to_array((clip_value,)),
                                          dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Cast()(reciprocal(scale), F.dtype(grad))


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad


class TrainOneStepWithLossScale(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=False,
                 config=None):
        super(TrainOneStepWithLossScale,
              self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.config = config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.learning_rate = self.optimizer.learning_rate
        self.global_step = self.optimizer.global_step
        self.default_lr = Tensor([0.0], dtype=mstype.float32)
        self.enable_global_norm = enable_global_norm
        # this config seems deleted
        self.enable_offload = False
        self.clip_value = Tensor([1.0], dtype=mstype.float32)
        if self.enable_offload:
            self.clip = GlobalNorm(self.weights, config)
        else:
            self.clip = ClipByGlobalNorm(self.weights, config, clip_norm=10.0)
        self.cast = P.Cast()

    def construct(self,
                  query_tensors, response_tensors, logprobs, values, rewards,
                  advantages, returns, pretrain_ids, loss_mask, attention_mask):
        """Defines the computation performed."""
        lr = self.learning_rate(self.global_step)
        weights = self.weights
        # Forward process
        loss = self.network(query_tensors, response_tensors, logprobs, values, rewards,
                            advantages, returns, pretrain_ids, loss_mask, attention_mask)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        # Backward process using loss scale
        grads = self.grad(self.network,
                          weights)(query_tensors, response_tensors, logprobs, values, rewards,
                                   advantages, returns, pretrain_ids, loss_mask, attention_mask, scaling_sens_filled)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(grad_scale, scaling_sens), grads)
        clip_value = self.clip_value
        if self.enable_global_norm:
            grads, clip_value = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        # Check whether overflow
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # If overflow, surpass weights update
        # if not, update weights
        if not overflow:
            if self.enable_offload:
                self.optimizer(grads, clip_value)
            else:
                self.optimizer(grads)
        return loss, lr, cond, scaling_sens.value()


class TrainPipelineWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, config, scale_update_cell=None, enable_global_norm=True):
        super(TrainPipelineWithLossScaleCell, self).__init__(auto_prefix=False)
        self.config = config
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.learning_rate = self.optimizer.learning_rate
        self.global_step = self.optimizer.global_step
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.reshape = P.Reshape()
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
        self.clip = ClipByGlobalNorm(self.weights, self.config)
        self.micro_size = config.parallel_config.micro_batch_num
        self.opt_shard = _get_enable_parallel_optimizer()
        self.enable_offload = False
        self.clip_value = Tensor([1.0], dtype=mstype.float32)
        if self.enable_offload:
            self.clip = GlobalNorm(self.weights, config)
        else:
            self.clip = ClipByGlobalNorm(self.weights, config)

    @C.add_flags(has_effect=True)
    def construct(self,
                  query_tensors, response_tensors, logprobs, values, rewards, advantages,
                  returns, pretrain_ids, loss_mask, attention_mask, sens=None):
        """Defines the computation performed."""
        lr = self.learning_rate(self.global_step)
        weights = self.weights
        loss = self.network(query_tensors, response_tensors, logprobs, values, rewards, advantages,
                            returns, pretrain_ids, loss_mask, attention_mask)
        if sens is None:
            scaling_sens = self.loss_scale
            scaling_sens = self.reshape(scaling_sens, (1,))
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        status_clear = self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(query_tensors, response_tensors, logprobs, values, rewards, advantages,
                                                 returns, pretrain_ids, loss_mask, attention_mask,
                                                 self.cast(scaling_sens / self.micro_size, mstype.float32))
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        loss = F.depend(loss, status_clear)
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        grads = F.depend(grads, cond)
        # apply grad reducer on grads
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)
        clip_value = self.clip_value
        if self.enable_global_norm:
            grads, _ = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            if self.enable_offload:
                self.optimizer(grads, clip_value)
            else:
                self.optimizer(grads)
        return loss, lr, overflow, scaling_sens.value()


class TrainOneStepWithLossScaleGRPO(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=False,
                 config=None):
        super(TrainOneStepWithLossScaleGRPO,
              self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.config = config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.learning_rate = self.optimizer.learning_rate
        self.global_step = self.optimizer.global_step
        self.default_lr = Tensor([0.0], dtype=mstype.float32)
        self.enable_global_norm = enable_global_norm
        # this config seems deleted
        self.enable_offload = False
        self.clip_value = Tensor([1.0], dtype=mstype.float32)
        if self.enable_offload:
            self.clip = GlobalNorm(self.weights, config)
        else:
            self.clip = ClipByGlobalNorm(self.weights, config, clip_norm=10.0)
        self.cast = P.Cast()

    def construct(self,
                  prompt_completion_ids, responses_mask,
                  ref_per_token_logps, advantages,
                  actual_sequence_length, sample_index, sample_valid_length):
        """Defines the computation performed."""
        lr = self.learning_rate(self.global_step)
        weights = self.weights
        # Forward process
        loss = self.network(prompt_completion_ids, responses_mask,
                            ref_per_token_logps, advantages,
                            actual_sequence_length, sample_index, sample_valid_length)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        # Backward process using loss scale
        grads = self.grad(self.network,
                          weights)(prompt_completion_ids, responses_mask,
                                   ref_per_token_logps, advantages,
                                   actual_sequence_length, sample_index, sample_valid_length,
                                   scaling_sens_filled)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(grad_scale, scaling_sens), grads)
        clip_value = self.clip_value
        if self.enable_global_norm:
            grads, clip_value = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        # Check whether overflow
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # If overflow, surpass weights update
        # if not, update weights
        if not overflow:
            if self.enable_offload:
                self.optimizer(grads, clip_value)
            else:
                self.optimizer(grads)
        return loss, lr, cond, scaling_sens.value()


class TrainPipelineWithLossScaleCellGRPO(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, config, scale_update_cell=None, enable_global_norm=True):
        super(TrainPipelineWithLossScaleCellGRPO, self).__init__(network, optimizer, scale_update_cell)
        if isinstance(scale_update_cell, (int, float)):
            scale_update_cell = Tensor(scale_update_cell)
        self.config = config
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.learning_rate = self.optimizer.learning_rate
        self.global_step = self.optimizer.global_step
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.reshape = P.Reshape()
        self.loss_scaling_manager = scale_update_cell
        # 8-bit status param for get_overflow_status func
        self.status = Tensor([0] * 8, mstype.int32)
        if isinstance(scale_update_cell, nn.Cell):
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
        elif isinstance(scale_update_cell, Tensor):
            if scale_update_cell.shape == (1,) or scale_update_cell.shape == ():
                self.loss_scale = Parameter(scale_update_cell, name='loss_scale')
            else:
                raise ValueError("The shape of 'scale_sense' must be (1,) or (), but got {}"
                                 .format(scale_update_cell.shape))
        self.clip = ClipByGlobalNorm(self.weights, self.config)
        self.micro_size = config.parallel_config.micro_batch_num
        self.opt_shard = _get_enable_parallel_optimizer()
        self.enable_offload = False
        self.clip_value = Tensor([1.0], dtype=mstype.float32)
        if self.enable_offload:
            self.clip = GlobalNorm(self.weights, config)
        else:
            self.clip = ClipByGlobalNorm(self.weights, config)

    @C.add_flags(has_effect=True)
    def construct(self,
                  prompt_completion_ids, responses_mask,
                  ref_per_token_logps, advantages,
                  actual_sequence_length, sample_index, sample_valid_length,
                  sens=None):
        """Defines the computation performed."""
        lr = self.learning_rate(self.global_step)
        weights = self.weights
        loss = self.network(prompt_completion_ids, responses_mask,
                            ref_per_token_logps, advantages,
                            actual_sequence_length, sample_index, sample_valid_length)
        if sens is None:
            scaling_sens = self.loss_scale
            scaling_sens = self.reshape(scaling_sens, (1,))
        else:
            scaling_sens = sens
        grads = self.grad(self.network, weights)(prompt_completion_ids, responses_mask,
                                                 ref_per_token_logps, advantages,
                                                 actual_sequence_length, sample_index, sample_valid_length,
                                                 self.cast(scaling_sens / self.micro_size, mstype.float32))
        # apply grad reducer on grads
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)
        clip_value = self.clip_value
        if self.enable_global_norm:
            grads, _ = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        cond = self.get_overflow_status(self.status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            if self.enable_offload:
                self.optimizer(grads, clip_value)
            else:
                self.optimizer(grads)
        return loss, lr, overflow, scaling_sens.value()
