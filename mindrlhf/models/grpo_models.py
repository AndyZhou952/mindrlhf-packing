# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
# limitations under the License
# ============================================================================
"""GRPO model"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.ops import operations as P
from mindrlhf.utils.generator import GeneratorMixin
from mindformers.models.utils import lazy_inline
from .base_model import BaseModel

__all__ = [
    "GRPOModel",
    "CausalLMHybrid",
]

class CausalLMHybrid(BaseModel):
    """
    CausalLMHybrid
    """
    def __init__(self, model_config, grpo_config, is_training=True):
        super(CausalLMHybrid, self).__init__()
        if not is_training:
            model_config.dropout_rate = 0.0
        self.model_config = model_config
        self.grpo_config = grpo_config
        self.select_actor_model(model_config)
        self.lm_head.pipeline_stage = model_config.parallel_config.pipeline_stage - 1
        dp = model_config.parallel_config.data_parallel
        mp = model_config.parallel_config.model_parallel
        cp = model_config.parallel_config.context_parallel

        self.vocab_size = model_config.vocab_size
        self.chunk_size = grpo_config.chunk_size
        self.seq_length = grpo_config.seq_length
        self.cast = P.Cast()
        self.all_ones_attention_mask = Tensor(
            np.ones((1, 1, self.seq_length)), mstype.float32
        )

        self.squeeze = P.Squeeze(axis=-1).shard(((dp, 1, 1),))
        self.squeeze_no_shard_1 = P.Squeeze(axis=-1).shard(((1, 1),))
        self.squeeze_no_shard_2 = P.Squeeze(axis=-1).shard(((1, 1, 1),))
        self.unsqueeze = P.ExpandDims().shard(((dp*mp*cp,),))
        self.unsqueeze_2 = P.ExpandDims().shard(((dp, 1),))
        self.reshape = P.Reshape()
        self.gather = P.Gather().shard(((dp, mp), (1,),))
        self.gatherd = P.GatherD().shard(((dp*mp*cp, 1), (dp*mp*cp, 1)))
        self.gatherd_2 = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.logsoftmax = P.LogSoftmax().shard(((dp*mp*cp, 1),))
        self.logsoftmax_1 = P.LogSoftmax().shard(((dp*mp*cp, 1),))
        self.logsoftmax_2 = P.LogSoftmax().shard(((dp, 1, 1),))
        self.expaned = P.ExpandDims().shard(((dp, mp),))

        self.pow = P.Pow().shard(((dp, 1), ()))
        self.argmax_no_shard = P.Argmax(-1).shard(((1, 1),))
        self.argmax = P.Argmax(-1).shard(((dp, mp),))
        self.add_shard = P.Add().shard(((1, 1, 1), ()))

        self.pad_logits = ops.Pad(((0, 0), (1, 0), (0, 0))).shard(((dp*mp, 1, 1),))
        self.pad_samples = ops.Pad(((0, 0), (1, 0))).shard(((dp*mp, 1),))

        self.expaned = P.ExpandDims().shard(((dp, mp),))
        self.dp = dp

    def offset_actual_seq_length(self, data, offset):
        bs = data.shape[0] // self.dp
        n = data.shape[1]
        data_type = data.dtype
        data = data.reshape((self.dp, bs, n))
        offsets = self.cast(ops.range(0, bs * offset, offset).reshape((1, bs, 1)), data_type)
        data = data + offsets
        actual_seq_lenth = self.cast(ops.reshape(data, (-1,)), data_type)
        return actual_seq_lenth

    def process_logits2(
            self, logits, current_index=None, is_first_iteration=False, use_past=False
    ):
        r"""
        process_logits2
        """
        logits = logits.reshape(-1, logits.shape[-1])
        if use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(
                -1,
            )
            logits = self.gather(logits, index, 0)
        top_token_id = self.argmax_no_shard(logits)
        top_token_id = top_token_id.view(-1, 1)
        return top_token_id

    def logprobs_of_labels(self, logits, samples):
        """
        Calculate the log value of the label
        """
        bs, seq_len = samples.shape
        logprobs = self.logsoftmax(logits)  # [bs*seq_len, vocab_size]
        samples = self.expaned(samples, -1)
        samples = self.reshape(samples, (bs*seq_len, -1))
        logprobs = self.squeeze_no_shard_1(
            self.gatherd(logprobs, -1, samples)
        )  # [bs, seq_len]
        logprobs = self.reshape(logprobs, (bs, seq_len))
        return logprobs  # [bs, seq_len-1]

    def construct(
            self,
            # inputs for the llm
            input_ids,
            input_position=None,
            batch_valid_length=None,
            slot_mapping=None,
            # inputs for `process_logits`
            is_first_iteration=False,
            use_past=False,
            # inputs for choosing the output branch
            samples=None,
            actual_seq_length=None,
            return_full_logit=False,
            is_ref=False
    ):
        """
        construct function for CausalLMHybrid
        """
        batch_size, seq_length = input_ids.shape

        if actual_seq_length is not None:
            if len(actual_seq_length.shape) > 1:
                bsz, _ = actual_seq_length.shape
                if bsz > 1:
                    actual_seq_length = self.offset_actual_seq_length(actual_seq_length, input_ids.shape[1])
        if self.model_type == "llama":
            if self.model.phase == "train":
                tokens = input_ids
            else:
                tokens = input_ids
                if batch_valid_length is None or slot_mapping is None:
                    bsz, seqlen = tokens.shape
                    batch_valid_length = ops.ones((bsz,), mstype.int32).reshape(-1)
                    slot_mapping = Tensor(np.ones(shape=tuple([bsz * seqlen])), mstype.int32)
            output_states = self.backbone(tokens, None, batch_valid_length, None, None, None, slot_mapping,
                                          None, None, None, None, None, actual_seq_length)
            logits_2d = self.lm_head(output_states)
        elif self.model_type == "deepseek_infer":
            tokens = input_ids
            if batch_valid_length is None or slot_mapping is None:
                bsz, seqlen = tokens.shape
                batch_valid_length = ops.ones((bsz,), mstype.int32).reshape(-1)
                slot_mapping = Tensor(np.ones(shape=tuple([bsz * seqlen])), mstype.int32)
            output_states = self.backbone(tokens, batch_valid_length=batch_valid_length,
                                          slot_mapping=slot_mapping)
            logits_2d = self.lm_head(output_states)
        elif self.model_type == "deepseek_training":
            tokens = input_ids
            # TODO: extra_loss=0.是否合理
            output_states, _ = self.backbone(tokens, extra_loss=0.)
            logits_2d = self.lm_head(output_states)
        else:
            raise NotImplementedError("only support {}".format(" ".join(self.model_type)))

        logits = self.reshape(logits_2d, (batch_size * seq_length, -1))
        # used in inference of make_experience and grpo
        if samples is not None:
            if is_ref:
                return logits
            logprobs_labels = self.logprobs_of_labels(logits, samples)
            return logprobs_labels
        # used in generate
        if not return_full_logit:
            outputs = self.process_logits2(
                logits, input_position, is_first_iteration, use_past
            )
            return outputs
        # used in pretrain loss
        logits = self.add_shard(logits, 0)
        return logits


class GRPOModel(nn.Cell, GeneratorMixin):
    """ GRPOModel """
    @lazy_inline
    def __init__(self, grpo_config, policy_model):
        super(GRPOModel, self).__init__()
        self.grpo_config = grpo_config
        self.beta = grpo_config.beta
        self.pad_token_id = Tensor(grpo_config.pad_token_id, mstype.int32)
        self.policy_model = policy_model

        self.concat = P.Concat(1)
        self.exp = P.Exp()
        self.gamma = 1
        self.lam = 0.95
        self.depend = P.Depend()
        self.dp = self.policy_model.dp
        self.cast = P.Cast()
        self.dump = P.TensorDump()

    def batch_unsorted_segment_sum(self, input_ids, segments_ids, num_segments):
        """
        GRPOModel
        batch unsorted_segment_sum

        Args:
            input_ids: shape (batch_size, seq_len)
            segment_ids: shape (batch_size, seq_len)
            num_segments: int
        Returns:
            shape (batch_size, num_segments)
        """
        bs, _ = input_ids.shape
        offsets = ops.arange(0, bs * num_segments, num_segments)
        # ensure segment_id uniqueness after reshape
        seg_off = segments_ids + offsets.view(bs, 1)
        flat_sum = ops.unsorted_segment_sum(
            input_ids.view(-1),
            seg_off.view(-1),
            bs * num_segments
        )
        return flat_sum.view(bs, num_segments)

    # pylint: disable=W0613
    def offset_actual_seq_length(self, data, offset):
        bs = data.shape[0] // self.dp
        n = data.shape[1]
        data_type = data.dtype
        data = data.reshape((self.dp, bs, n))
        offsets = self.cast(ops.range(0, bs * offset, offset).reshape((1, bs, 1)), data_type)
        data = data + offsets
        actual_seq_lenth = self.cast(ops.reshape(data, (-1,)), data_type)
        return actual_seq_lenth

    def construct(
            self,
            prompt_completion_ids, # [bs, seq_len]
            responses_mask,  # [bs, seq_len]
            ref_per_token_logps, # [bs, seq_len]
            advantages,  # [bs, seq_len]
            actual_seq_length,  # [bs, packed_sample_num]
            sample_index, #[bs, seq_len]
            sample_valid_len,  #[bs, packed_sample_num]
        ):
        """ construct function for GRPOModel """
        # bs, seq_len = prompt_completion_ids.shape
        pack_sample_num = sample_valid_len.shape[1]
        real_sample_num = ops.sum(sample_valid_len != 1, dtype=mstype.int32)

        input_ids = prompt_completion_ids[:, :-1]  # [bs, seq_len]
        samples = prompt_completion_ids[:, 1:]  # [bs, seq_len]
        actual_seq_length = self.offset_actual_seq_length(actual_seq_length, input_ids.shape[1])
        per_token_logps = self.policy_model(input_ids, None, None, None, False, False,
                                            samples, actual_seq_length, False, False) # [bs, seq_len]
        per_token_logps = per_token_logps * responses_mask
        per_token_kl = self.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1  # [bs, seq_len]
        per_token_loss = self.exp(per_token_logps - ops.stop_gradient(per_token_logps)) * advantages # [bs, seq_len]
        per_token_loss = - (per_token_loss - self.beta * per_token_kl)  # [bs, seq_len]
        masked_per_token_loss = per_token_loss * responses_mask  # [bs, seq_len-1]
        deno = self.batch_unsorted_segment_sum(masked_per_token_loss, sample_index, pack_sample_num)  # [bs, packed_sample_num]
        nume = sample_valid_len  #  [bs, packed_sample_num]
        loss = (deno/nume).sum()/real_sample_num
        return loss


class GRPOModelInfer(nn.Cell):
    def __init__(self, grpo_config, policy_model):
        super(GRPOModelInfer, self).__init__()
        self.grpo_model = GRPOModel(grpo_config, policy_model)

    def construct(self, *args, **kwargs):
        return self.grpo_model(*args, **kwargs)


class GRPOModelTrain(nn.Cell):
    def __init__(self, grpo_config, policy_model):
        super(GRPOModelTrain, self).__init__()
        self.grpo_model_train = GRPOModel(grpo_config, policy_model)

    def construct(self, *args, **kwargs):
        return self.grpo_model_train(*args, **kwargs)
