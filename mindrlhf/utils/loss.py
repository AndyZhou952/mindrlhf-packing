import math
import mindspore as ms
import numpy as np
from enum import Enum
from mindspore import Tensor, mint, ops
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindspore.nn.cell import Cell

__all__ = ['DPO_Loss', 'CompareLoss']

class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0


class RunningMoments:
    """
    Calculates the running mean and standard deviation of a data stream. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
    """
    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    def update(self, xs):
        """
        Updates running moments from batch's moments computed across ranks
        """

        xs_count = xs.numel()
        xs_var, xs_mean = ms.ops.var_mean(xs)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta ** 2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


def cap_exp(value, cap=-1):
    # Cap the exponent value below the upper-bound to avoid overflow, before calling mint.exp
    cap = get_exp_cap(value) if cap < 0 else cap
    return mint.exp(mint.clamp(value, max=cap))


def get_exp_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow.
    The formula is : log(value.dtype.max)
    E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.
    ```

    Args:
        value (`ms.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap.
            eg: direct calling exp(log(np.float32.max)) will result in inf
            so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = mint.zeros([1], dtype=value.dtype) + ms.Tensor(np.finfo(mstype.dtype_to_nptype(value.dtype)).max)
    vdtype_log_max = mint.log(vdtype_max)
    return mint.floor(vdtype_log_max * 10 ** decimal) / 10 ** decimal if decimal > 0 else vdtype_log_max


class DPO_Loss(Cell):
    def __init__(self, config):
        super(DPO_Loss, self).__init__()
        self.config = config
        self.reference_free = self.config.reference_free
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
        self.loss_type = self.config.loss_type
        self.label_smoothing = self.config.label_smoothing
        self.beta = self.config.beta
        self.f_divergence_type = self.config.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: self.config.f_alpha_divergence_coef}
        if self.loss_type == "bco_pair":
            self.running = RunningMoments()

    def construct(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
        chosen_logratios = policy_chosen_logps - self.config.reference_free * reference_chosen_logps
        rejected_logratios = policy_rejected_logps - self.reference_free * reference_rejected_logps

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = ms.Tensor([0], dtype=pi_logratios.dtype)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps
            logits = pi_logratios - ref_logratios
            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= mint.nn.functional.softplus(chosen_logratios) - mint.nn.functional.softplus(
                    rejected_logratios)
        if self.loss_type == "sigmoid":
            losses = -self.logsigmoid(self.config.beta * logits) * (1 - self.label_smoothing) - self.logsigmoid(
                -self.config.beta * logits) * self.label_smoothing
        elif self.loss_type == "robust":
            losses = (-self.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) + self.logsigmoid(
                -self.beta * logits) * self.label_smoothing) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                    self.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)) + (
                             -self.beta * logits).sigmoid() * (
                             self.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = mint.nn.functional.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = mint.cat((chosen_rewards, rejected_rewards), 0).mean()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -self.logsigmoid((self.beta * chosen_logratios) - delta) - self.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                    -self.logsigmoid(chosen_rewards)
                    - 0.5 * self.logsigmoid(-chosen_rewards)
                    - 0.5 * self.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = mint.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = mint.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                    -self.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - self.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            pi_logratios_sorted, _ = mint.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = mint.sort(ref_logratios, dim=0)
            delta = pi_logratios_sorted - ref_logratios_sorted
            losses = -self.logsigmoid(self.beta * delta) * (1 - self.label_smoothing) - self.logsigmoid(
                -self.beta * delta) * self.label_smoothing

        elif self.loss_type == "apo_zero":
            losses_chosen = 1 - self.sigmoid(self.beta * chosen_logratios)
            losses_rejected = self.sigmoid(self.beta * rejected_logratios)
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            losses_chosen = self.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - self.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        return logits, losses, chosen_rewards, rejected_rewards

class CompareLoss(nn.Cell):
    """
    Calculate the compare loss for reward model.

    Args:
        config (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **rewards** (Tensor) - Tensor of shape (B, S, 1). Data type must be float16 or float32. The output logits of
          the backbone.

        - **loss_mask** (Tensor) - Tensor of shape (B, S, 1). The loss mask of the rewards.

        - **end_ind** (Tensor) - Tensor of shape (B, ). end index of all tensors.

    Returns:
        The corresponding loss.
    """

    def __init__(self, config):
        super().__init__()
        dp = config.data_parallel
        mp = 1
        self.gatherd = P.GatherD()
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.slice_ind = P.StridedSlice().shard(((1,),))
        self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
        self.sub = P.Sub().shard(((dp, mp), (dp, mp)))

    def construct(self, rewards, loss_mask, end_ind):
        """Forward process"""
        bs = rewards.shape[0] // 2  # a sample has two bs responses
        seq_len = rewards.shape[-1]
        chosen_rewards = self.slice(rewards, (0, 0), (bs, seq_len), (1, 1))
        rejected_rewards = self.slice(rewards, (bs, 0), (2 * bs, seq_len), (1, 1))
        end_ind_chosen = self.slice_ind(end_ind, (0,), (bs,), (1,))
        end_ind_reject = self.slice_ind(end_ind, (bs,), (2 * bs,), (1,))
        temp = P.Concat()((end_ind_chosen, end_ind_reject))
        temp = temp.reshape((2, -1))
        temp = P.Cast()(temp, mstype.float16)
        end_ind_final, _ = P.max(temp, axis=0)
        temp = P.Cast()(temp, mstype.float16)
        end_ind_final = end_ind_final.reshape((-1, 1))
        end_ind_final = P.Cast()(end_ind_final, mstype.int32)
        loss_mask_final = loss_mask
        c_truncated_reward = self.mul(chosen_rewards, loss_mask_final)
        r_truncated_reward = self.mul(rejected_rewards, loss_mask_final)
        chosen_end_scores = self.gatherd(chosen_rewards, 1, end_ind_final - 1)
        reject_end_scores = self.gatherd(rejected_rewards, 1, end_ind_final - 1)
        compare_len = self.reduce_sum(P.cast(loss_mask_final, mstype.float32), -1)
        temp_loss = -self.log(P.sigmoid(self.sub(c_truncated_reward, r_truncated_reward)))
        loss = self.reduce_sum(self.mul(temp_loss, loss_mask_final), -1) / compare_len
        loss = loss.mean()
        return loss, chosen_end_scores, reject_end_scores
    