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
"""qwen2_5_7b_dpo models' APIs."""
import copy
import mindspore.common.dtype as mstype
from mindformers import LlamaConfig, LlamaForCausalLM
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.utils import lazy_inline
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindspore import Tensor, nn, ops
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode
from typing import Optional

__all__ = ['Qwen2_5_7BDPO']

class DPOLoss(nn.Cell):
    def __init__(self, config):
        super(DPOLoss, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.gatherd = P.GatherD()
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.slice = P.StridedSlice().shard(((1, 1),))  # ?
        self.slice_ind = P.StridedSlice().shard(((1,),))    # ?
        self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
        self.sub = P.Sub().shard(((dp, mp), (dp, mp)))
        self.log_softmax = P.LogSoftmax()
        self.squeeze = P.Squeeze(-1).shard(((1, 1, 1),))
        self.expand = P.ExpandDims().shard(((1, 1),))
        self.label_pad_token_id = config.pad_token_id
        self.average_log_prob = True
        self.reference_free = False
        self.log_sigmoid = nn.LogSigmoid()
        self.reduce_mean = P.ReduceMean()
        self.not_equal = P.NotEqual()
        self.beta = 0.1
        self.enable_force_redistribute = True
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)

    def _get_batch_logps(self, logits, labels, loss_mask=None):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with value of label_pad_token_id are ignored. Shape: (batch_size, seq_len)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if loss_mask is None:
            loss_mask = self.not_equal(labels, self.label_pad_token_id)
        # [bs, seq_len] -> [bs, seq_len]
        labels = self.mul(labels, loss_mask)
        # [bs, seq_len, vocab_size]
        log_probs = self.log_softmax(logits)
        # [bs, seq_len] -> [bs, seq_len, 1]
        index = self.expand(labels, -1)
        index = self.cast(index, mstype.int32)
        # [bs, seq_len, 1]
        per_token_logps = self.gatherd(log_probs, -1, index)
        # [bs, seq_len, 1] -> [bs, seq_len]
        per_token_logps = self.squeeze(per_token_logps)
        if self.average_log_prob:
            return self.reduce_sum(per_token_logps * loss_mask, -1) / self.reduce_sum(loss_mask, -1)
        else:
            return self.reduce_sum(per_token_logps * loss_mask, -1)

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, chosen_ref_logps, rejected_ref_logps):
        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        ref_log_ratios = chosen_ref_logps - rejected_ref_logps
        if self.reference_free:
            ref_log_ratios = 0
        logits = policy_log_ratios - ref_log_ratios
        losses = -self.log_sigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - chosen_ref_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - rejected_ref_logps)
        return losses, chosen_rewards, rejected_rewards

    def construct(self, policy_logits, policy_labels, loss_mask, chosen_ref_logps, rejected_ref_logps):
        # policy_logits: [bs, seq_len, vocab_size]
        # policy_labels: [bs, seq_len]
        # loss_mask: [bs, seq_len]
        # chosen_ref_logps: [bs,]
        # rejected_ref_logps: [bs,]
        # [bs,]
        all_logps = self._get_batch_logps(policy_logits, policy_labels, loss_mask)
        bs = all_logps.shape[0] // 2    # a sample has two bs responses (chosen and rejected)
        policy_chosen_logps = self.slice_ind(all_logps, (0,), (bs,), (1,))
        policy_rejected_logps = self.slice_ind(all_logps, (bs,), (2 * bs,), (1,))
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_ref_logps,
            rejected_ref_logps
        )
        if self.phase == "train":
            return losses
        return losses, chosen_rewards, rejected_rewards


class DPOCrossEntropy(CrossEntropyLoss):
    def __init__(self, parallel_config, **kwargs):
        super().__init__(parallel_config, **kwargs)
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.slice_3d = P.StridedSlice().shard(((dp, mp, 1),))
        self.slice_2d = P.StridedSlice().shard(((dp, mp),))

    def construct(self, logits, label, input_mask):
        bs, seq_len, vocab_size = logits.shape    # a sample has two bs responses (chosen and rejected)
        policy_chosen_logps = self.slice_3d(logits, (0, 0, 0), (bs // 2, seq_len, vocab_size), (1, 1, 1))
        label = self.slice_2d(label, (0, 0), (bs // 2, seq_len), (1, 1))
        input_mask = self.slice_2d(input_mask, (0, 0), (bs // 2, seq_len), (1, 1))
        return super().construct(policy_chosen_logps.reshape((-1, policy_chosen_logps.shape[-1])), label.reshape((-1,)), input_mask.reshape((-1,)))

@MindFormerRegister.register(MindFormerModuleType.LOSS)
class DPOLossV2(nn.Cell):
    def __init__(self, config):
        super(DPOLossV2, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.gatherd = P.GatherD()
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.slice = P.StridedSlice().shard(((1, 1),))  # ?
        self.slice_ind = P.StridedSlice().shard(((1,),))    # ?
        self.slice_mask = P.StridedSlice().shard(((1, 1),))
        self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
        self.sub = P.Sub().shard(((dp, mp), (dp, mp)))
        self.log_softmax = P.LogSoftmax()
        self.squeeze = P.Squeeze(-1).shard(((1, 1, 1),))
        self.expand = P.ExpandDims().shard(((1, 1),))
        self.label_pad_token_id = config.pad_token_id
        self.average_log_prob = True
        self.reference_free = False
        self.log_sigmoid = nn.LogSigmoid()
        self.not_equal = P.NotEqual()
        # for cal reward
        self.beta = config.beta
        self.enable_force_redistribute = True
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)

    def _get_batch_logps(self, logits, labels, loss_mask=None):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with value of label_pad_token_id are ignored. Shape: (batch_size, seq_len)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if loss_mask is None:
            loss_mask = self.not_equal(labels, self.label_pad_token_id)
        # [bs, seq_len] -> [bs, seq_len]
        labels = self.mul(labels, loss_mask)
        # [bs, seq_len, vocab_size]
        log_probs = self.log_softmax(logits)
        # [bs, seq_len] -> [bs, seq_len, 1]
        index = self.expand(labels, -1)
        index = self.cast(index, mstype.int32)
        # [bs, seq_len, 1]
        per_token_logps = self.gatherd(log_probs, -1, index)
        # [bs, seq_len, 1] -> [bs, seq_len]
        per_token_logps = self.squeeze(per_token_logps)
        if self.average_log_prob:
            return self.reduce_sum(per_token_logps * loss_mask, -1) / self.reduce_sum(loss_mask, -1)
        else:
            return self.reduce_sum(per_token_logps * loss_mask, -1)

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_mask):
        bs, seq_len = loss_mask.shape
        if self.average_log_prob:
            policy_chosen_logps_avg = policy_chosen_logps
        else:
            chosen_loss_mask = self.slice_mask(loss_mask, (0, 0), (bs // 2, seq_len), (1, 1))
            chosen_valid_len = self.reduce_sum(chosen_loss_mask, -1)
            policy_chosen_logps_avg = policy_chosen_logps / chosen_valid_len

        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        ref_log_ratios = ref_chosen_logps - ref_rejected_logps
        if self.reference_free:
            ref_log_ratios = 0
        logits = policy_log_ratios - ref_log_ratios
        losses = -self.log_sigmoid(self.beta * logits)  # if logits is very large, the losses would be nan
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        return losses, chosen_rewards, rejected_rewards, policy_chosen_logps_avg

    def construct(self, policy_logits, policy_labels, chosen_loss_mask, rejected_loss_mask, ref_chosen_logps, ref_rejected_logps):
        # policy_logits: [bs, seq_len, vocab_size]
        # policy_labels: [bs, seq_len]
        # loss_mask: [bs, seq_len]
        # ref_chosen_logps: [bs,]
        # ref_rejected_logps: [bs,]
        # [bs,]
        loss_mask = ops.concat((chosen_loss_mask, rejected_loss_mask), axis=0)
        all_logps = self._get_batch_logps(policy_logits, policy_labels, loss_mask)

        bs = all_logps.shape[0] // 2    # a sample has two bs responses (chosen and rejected)
        policy_chosen_logps = self.slice_ind(all_logps, (0,), (bs,), (1,))
        policy_rejected_logps = self.slice_ind(all_logps, (bs,), (2 * bs,), (1,))

        if self.average_log_prob:
            ref_chosen_logps = ref_chosen_logps / self.reduce_sum(chosen_loss_mask, -1)
            ref_rejected_logps = ref_rejected_logps / self.reduce_sum(rejected_loss_mask, -1)

        dpo_loss, chosen_rewards, rejected_rewards, policy_chosen_logps_avg = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_mask
        )
        sft_loss = -policy_chosen_logps_avg
        
        if self.phase == "train":
            return dpo_loss, sft_loss
        return dpo_loss, sft_loss, chosen_rewards, rejected_rewards

@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Qwen2_5_7BDPO(LlamaForCausalLM):
    r"""
        Provide qwen2.5_7B training loss or logits through network.
        Args:
            config (LlamaConfig): The config of qwen1.5_7B model.

        Inputs:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
              prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.

        Returns:
            Tensor, the loss or logits of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> from mindrlhf.models.qwen2_5.qwen_dpo import Qwen2_5_7BDPO
            >>> config = LlamaConfig(batch_size=2)
            >>> network = Qwen2_5_7BDPO(config=config)
        """

    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(Qwen2_5_7BDPO, self).__init__(config)
        _check_config(config.parallel_config)
        self.config = config
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.dtype = config.compute_dtype

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.print = ops.Print()

        vocab_size = config.vocab_size
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.dpo_loss = DPOLossV2(config)
        else:
            loss_parallel_config = copy.deepcopy(config)
            loss_parallel_config.parallel_config.model_parallel = dp * mp
            loss_parallel_config.parallel_config.data_parallel = 1
            if dp >= 32 and dp % 8 == 0: # For large scale training
                loss_parallel_config.parallel_config.model_parallel = 8
                loss_parallel_config.parallel_config.data_parallel = dp * mp // 8
            self.dpo_loss = DPOLossV2(loss_parallel_config)

        self.alpha = config.alpha
        self.beta = config.beta
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.sft_loss = DPOCrossEntropy(parallel_config=config.parallel_config)
        else:
            loss_parallel_config = copy.deepcopy(config.parallel_config)
            loss_parallel_config.model_parallel = dp * mp
            loss_parallel_config.data_parallel = 1
            if dp >= 32 and dp % 8 == 0: # For large scale training
                loss_parallel_config.model_parallel = 8
                loss_parallel_config.data_parallel = dp * mp // 8
            self.sft_loss = DPOCrossEntropy(parallel_config=loss_parallel_config)

    # pylint: disable=W0613
    def construct(self, chosen_input_ids, chosen_labels=None, chosen_loss_mask=None, 
                    chosen_ref_logps=None, rejected_input_ids=None, rejected_labels=None,
                    rejected_loss_mask=None, rejected_ref_logps=None,
                  input_position=None, position_ids=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """
        construct function for qwen2_5
        """
        if self.training:
            input_ids = ops.concat((chosen_input_ids, rejected_input_ids), axis=0)
            labels = ops.concat((chosen_labels, rejected_labels), axis=0)
        else:
            input_ids = chosen_input_ids
            labels = chosen_labels
        bsz, ori_seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, ori_seqlen - 1), (1, 1))
            chosen_loss_mask = self.slice(chosen_loss_mask, (0, 1), (bsz, ori_seqlen), (1, 1))
            rejected_loss_mask = self.slice(rejected_loss_mask, (0, 1), (bsz, ori_seqlen), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        
        output = self.model(tokens, batch_valid_length, block_tables, slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)
        
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, ori_seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, ori_seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            return logits, tokens, input_mask

        if logits.ndim <= 2:
            logits = self.reshape(logits, (bsz, tokens.shape[1], logits.shape[-1]))
        policy_logits = self.cast(logits, mstype.float32)
        dpo_loss, sft_loss = self.dpo_loss(policy_logits, labels, chosen_loss_mask, rejected_loss_mask, chosen_ref_logps.reshape((-1,)), rejected_ref_logps.reshape((-1,)))
        return  dpo_loss + self.alpha * sft_loss
