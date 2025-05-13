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
""" GRPO Trainer """

# python
import os
import time
from dataclasses import asdict
import numpy as np

# mindspore
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.dataset import MindDataset
from mindspore.common.api import _pynative_executor
from mindspore import Tensor, mint
from mindspore.communication import get_rank
from mindspore.mindrecord import FileWriter

# mindformers
from mindformers import logger
from mindformers.models.llama import LlamaTokenizerFast
from mindformers import MindFormerConfig
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.utils.tensorboard import get_tensorboard_writer, _set_tensorboard_writer

from mindrlhf.trainer.packing_utils import pack_grpo_data, construct_inputs_packing
from mindrlhf.utils import (transfer_from_str_to_bool, yaml_to_dataclass, set_perf_stats, print_perf_stat,
                            convert_index_json_total, save_prompt_completions_data, MetricData, get_dp_rank)
# mindrlhf
from mindrlhf.reward.reward_fn import accuracy_reward, format_reward, reward_func_from_jiaoda
from mindrlhf.worker.infer_worker import InferWorker
from mindrlhf.worker.ref_worker import RefWorker
from mindrlhf.worker.train_worker import TrainWorker
from mindrlhf.worker.transform_worker import TransformWorker
import mindrlhf.utils.reshard_optimizer as reshard_optimizer
from mindrlhf.worker.worker import GRPOData
from mindrlhf.configs.grpo_configs import GRPOConfig, VllmMode
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer

class GRPOTrainer:
    """ GRPO Trainer """

    def __init__(self, args=None):
        """Initialize"""
        self.args = args
        self._init_grpo_configs(args)
        self._init_reward_fn(args)

        # ================== Initial Tensorboard ==================
        if hasattr(args, 'tensorboard_dir') and args.tensorboard_dir:
            args.tensorboard_dir = os.path.join(args.tensorboard_dir, f"rank_{get_rank()}")
            _set_tensorboard_writer(args)
        self.tensor_writer = get_tensorboard_writer()
        setattr(self.args, 'tensor_writer', self.tensor_writer)
        self.make_exp_step = 0

        logger.info("GRPOTrainer: start init workers")
        self.infer = InferWorker(grpo_config=self.grpo_config,
                                 sft_path_infer=self.sft_path_infer,
                                 args=self.args)
        # grpo_config infer and train share
        self.grpo_config = self.infer.get_updated_grpo_config()
        self.infer_dp = self.infer.get_infer_dp()
        self._init_grpo_infer_dataset()

        self.ref = RefWorker(grpo_config=self.grpo_config,
                             sft_path_ref=self.sft_path_ref,
                             args=self.args)
        self.ref_dp = self.ref.get_ref_dp()
        self.train = TrainWorker(grpo_config=self.grpo_config,
                                 sft_path_train=self.sft_path_train,
                                 args=self.args)
        logger.info(f"config of sft_model_config_train {self.train.sft_model_config_train}")
        logger.info("GRPOTrainer: finish init workers")

        self.reshard_optimizer = None
        self.reshard_mem_opt_level = self.grpo_config.reshard_mem_opt_level
        if self.reshard_mem_opt_level not in [0, 1]:
            raise ValueError(f"reshard_mem_opt_level can only be 0 or 1, but got {self.reshard_mem_opt_level}")
        # rename parameters in safetensors
        if args.load_sft_checkpoint_train and args.load_ckpt_format == "safetensors":
            self.rename_safetensors_weights(args)

        self._compile()
        self.transform = TransformWorker(self.grpo_config, self.train.sft_model_config_train,
                                         self.train.model(), self.infer.model(), self.ref.model())
        self._load_checkpoint()
        if not args.load_sft_checkpoint_infer:
            self.transform.reshard_params(0)

    def _init_grpo_configs(self, args):
        """ init grpo configs """
        logger.info(f"GRPOTrainer: _init_grpo_configs {args} in main task")
        use_parallel = transfer_from_str_to_bool(args.use_parallel)
        # init grpo config
        grpo_config = yaml_to_dataclass(args.config, GRPOConfig)
        grpo_config.mind_dataset_dir = args.mind_dataset_dir
        grpo_config.save_data_file = args.save_data_file
        grpo_config.save_ckpt_dir = args.save_ckpt_dir
        grpo_config.save_strategy_dir = args.save_strategy_dir
        grpo_config.align_type = "rlhf_stages"
        grpo_config.use_parallel = use_parallel
        set_perf_stats(grpo_config)
        if grpo_config.use_vllm not in range(len(VllmMode)):
            logger.warning(f"use_vllm should be 0, 1 or 2, but got {grpo_config.use_vllm}. Reset to 0.")
            grpo_config.use_vllm = 0
        grpo_config.use_vllm = VllmMode(grpo_config.use_vllm)
        logger.info(f"vllm mode: {grpo_config.use_vllm}, hf_config_path: {grpo_config.hf_config_path}")

        # for worker
        if args.custom_model_name == "qwen":
            self.tokenizer = Qwen2Tokenizer(
                args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False)
        elif args.custom_model_name == "deepseek":
            self.tokenizer = LlamaTokenizerFast(
                tokenizer_file=args.tokenizer_path, add_bos_token=False, add_eos_token=False
            )
        elif args.custom_model_name == "llama":
            sft_config_infer = MindFormerConfig(args.sft_path_infer)
            sft_config_infer.processor.tokenizer.tokenizer_file = args.vocab_path
            self.tokenizer = build_tokenizer(sft_config_infer.processor.tokenizer)
        else:
            raise ValueError(
                f"model_name should in ['qwen', 'deepseek'], but get {args.custom_model_name}")
        self.grpo_config = grpo_config
        self.args.use_parallel = use_parallel
        self.use_parallel = use_parallel
        self.sft_path_infer = args.sft_path_infer
        self.sft_path_ref = args.sft_path_ref
        self.sft_path_train = args.sft_path_train

    def _init_reward_fn(self, args):
        """ init reward function """
        logger.info("GRPOTrainer: _init_reward_fn")
        if args.reward_funcs:
            reward_funcs_list = args.reward_funcs
        else:
            reward_funcs_list = ["accuracy_reward", "format_reward"]
        if args.reward_weights:
            reward_weights = args.reward_weights
        else:
            reward_weights = [1.0, 1.0]

        reward_funcs = []
        for reward_func_str in reward_funcs_list:
            if reward_func_str == "accuracy_reward":
                reward_funcs.append(accuracy_reward)
            elif reward_func_str == "format_reward":
                reward_funcs.append(format_reward)
            elif reward_func_str == "reward_func_from_jiaoda":
                reward_funcs.append(reward_func_from_jiaoda)
            else:
                raise ValueError(f"Unsupported reward function {reward_func_str}")
        self.reward_funcs = reward_funcs

        # Reward weights
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(len(reward_weights))}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        self.reward_weights = np.array(reward_weights, dtype=np.float32)

    def _init_grpo_infer_dataset(self):
        """
        Build dataset for generating.
        """
        logger.info(
            "GRPOTrainer: _init_grpo_infer_dataset, dataset dir {self.mind_dataset_dir}")
        self.mind_dataset_dir = self.grpo_config.mind_dataset_dir
        if self.mind_dataset_dir is not None:
            columns_to_project = ["prompt_ids", "pretrain_ids", "loss_mask"]
            ms.dataset.config.set_seed(2023)
            dataset = MindDataset(self.mind_dataset_dir).project(columns=columns_to_project)
            self.prompt_dataset = dataset
            self.prompt_dataloader = dataset.take()
            bs = self.grpo_config.chunk_size * self.infer_dp
            self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=bs, drop_remainder=True)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
            self.step_num = self.prompt_dataloader.get_dataset_size() // self.grpo_config.num_rollouts
        else:
            logger.info("In main task, there is not dataset for making experience")

    def _compile(self):
        """
        compile model
        """
        if self.grpo_config.enable_reshard_optimizer:
            logger.info("Reshard Optimizer is enabled")

            train_parallel_config = MindFormerConfig(
                self.sft_path_train
            ).parallel_config
            infer_parallel_config = MindFormerConfig(
                self.sft_path_infer
            ).parallel_config

            self.reshard_optimizer = reshard_optimizer.ReshardOptimizer(
                src_parallel=reshard_optimizer.Parallel(
                    dp=train_parallel_config["data_parallel"],
                    tp=train_parallel_config["model_parallel"],
                    pp=train_parallel_config["pipeline_stage"],
                ),
                dst_parallel=reshard_optimizer.Parallel(
                    dp=infer_parallel_config["data_parallel"],
                    tp=infer_parallel_config["model_parallel"],
                    pp=infer_parallel_config["pipeline_stage"],
                ),
            )
            reshard_optimizer.OPT_COMMUNICATION_GROUPS = self.reshard_optimizer.opt_communication_groups

        start_time = time.time()
        self.infer.generate_strategy(self.reshard_optimizer)
        self.ref.compile()
        self.train.compile()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "GRPOTrainer compile")

    def _load_checkpoint(self):
        start_time = time.time()
        self.infer.load_checkpoint()
        self.ref.load_checkpoint()
        self.train.load_checkpoint()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "GRPOTrainer load checkpoint")

    def _get_batch(self, num_rollouts):
        """ get batch """
        full_batch = None
        for _ in range(num_rollouts):
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                ms.dataset.config.set_seed(2023)
                self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
                batch = next(self.prompt_iterator)
            if full_batch is None:
                full_batch = batch
            else:
                full_batch = (
                    mint.cat((full_batch[0], batch[0])),
                    mint.cat((full_batch[1], batch[1]))
                )
        return full_batch

    def _split_for_data_parallel(self, batch_inputs, data_parallel_size):
        """
        split batch_inputs for data parallel
        """
        split_size = (batch_inputs.shape[0] // data_parallel_size)
        dp_rank_id = get_dp_rank(data_parallel_size)

        start = dp_rank_id * split_size
        stop = (dp_rank_id + 1) * split_size
        batch_inputs_for_this_rank = batch_inputs[start:stop]
        return batch_inputs_for_this_rank

    def _remove_right_padding(self, token_ids, padding_token=0):
        """ remove_right_padding """
        trimmed_sequences = []
        for seq in token_ids:
            # 将序列转换为列表以处理不同输入类型（如numpy数组）
            seq_list = list(seq)
            # 逆序查找第一个非填充标记的位置
            last_non_pad = next((i for i in reversed(
                range(len(seq_list))) if seq_list[i] != padding_token), None)
            # 截断右侧填充
            if last_non_pad is not None:
                trimmed_sequences.append(seq_list[:last_non_pad + 1])
            else:
                trimmed_sequences.append([])  # 全为填充时返回空列表
        return trimmed_sequences

    def _remove_left_padding(self, token_ids, padding_token=0):
        """ remove_left_padding """
        trimmed_sequences = []
        for seq in token_ids:
            # 将序列转换为列表以处理不同输入类型（如numpy数组）
            seq_list = list(seq)
            # 顺查找第一个非填充标记的位置
            last_non_pad = next(
                (i for i in range(len(seq_list)) if seq_list[i] != padding_token), None)
            # 截断左侧填充
            if last_non_pad is not None:
                trimmed_sequences.append(seq_list[last_non_pad:])
            else:
                trimmed_sequences.append([])  # 全为填充时返回空列表
        return trimmed_sequences

    def _construct_inputs_for_ref_model(self, right_padding_responses, responses_mask, left_padding_prompts,
                                        prompts_mask, ref_model_batch_size=None, idx=None):
        """ construct inputs for reference model """
        if ref_model_batch_size:
            start_idx = idx * ref_model_batch_size
            end_idx = (idx + 1) * ref_model_batch_size
            right_padding_responses = right_padding_responses[start_idx: end_idx]
            responses_mask = responses_mask[start_idx: end_idx]
            left_padding_prompts = left_padding_prompts[start_idx: end_idx]
            prompts_mask = prompts_mask[start_idx: end_idx]

        is_eos = right_padding_responses == self.tokenizer.eos_token_id
        eos_idx = np.full(is_eos.shape[0], is_eos.shape[1], dtype=int)
        eos_idx[is_eos.any(axis=1)] = np.argmax(is_eos.astype(int), axis=1)[is_eos.any(axis=1)]
        sequence_indices = np.arange(is_eos.shape[1])
        responses_eos_mask = (sequence_indices <= eos_idx[:, None]).astype(int)
        responses_mask = responses_mask * responses_eos_mask

        prompt_completion_ids = np.concatenate((left_padding_prompts, right_padding_responses), axis=1)
        prompts_mask = np.concatenate((prompts_mask, np.zeros_like(responses_mask, dtype=np.int32)), axis=1)
        responses_mask = np.concatenate(
            (np.zeros_like(left_padding_prompts, dtype=np.int32), responses_mask), axis=1)
        attention_mask = prompts_mask + responses_mask

        # Generate outputs.
        attention_mask_tensor = Tensor(attention_mask, dtype=ms.int32)
        return prompt_completion_ids, attention_mask_tensor, responses_mask, prompts_mask

    def compute_advantages(self, rewards, eps=1e-4):
        mean_grouped_rewards = rewards.mean()
        if rewards.shape[0] == 1:
            std_grouped_rewards = rewards.std()
        else:
            std_grouped_rewards = rewards.std(ddof=1)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + eps)
        logger.info(f"mean_grouped_rewards: \n {mean_grouped_rewards}")

        return advantages, mean_grouped_rewards

    def _make_experience(self, num_rollouts: int = 1, num_generations: int = 16, step: int = 0):
        """ make experience """
        ep_begin_time = time.time()
        logger.info("Make experience begin at {} \n------------------------------- "
                    .format(time.strftime('%H:%M:%S', time.localtime(ep_begin_time))))
        logger.info(f"Generate {num_generations} times")
        if self.infer.use_vllm == VllmMode.ORIGIN:
            self.infer.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        else:
            self.infer.grpo_model_infer.grpo_model.policy_model.set_train(False)
        self.ref.ref_model.model.set_train(False)

        grpo_rl_elements = []
        all_mean_grouped_rewards = []
        all_elements_completion_len = []

        batch = self._get_batch(num_rollouts)
        prompt_tensors = Tensor(batch[0], mstype.int32).asnumpy()
        solution_ids = Tensor(batch[1], mstype.int32).asnumpy()
        prompt_tensors_full = prompt_tensors
        repeat_solution_ids = solution_ids
        for i in range(num_generations - 1):
            prompt_tensors_full = np.concatenate((prompt_tensors_full, prompt_tensors))
            repeat_solution_ids = np.concatenate((repeat_solution_ids, solution_ids))
        input_ids_numpy = self._split_for_data_parallel(prompt_tensors_full, self.infer_dp)
        solution_ids = self._remove_right_padding(repeat_solution_ids, padding_token=self.grpo_config.pad_token_id)
        solution = self.tokenizer.decode(solution_ids, skip_special_tokens=True)
        for i in range(len(solution)):
            solution[i] = "$" + solution[i] + "$"
        reward_kwargs = {"solution": solution}
        logger.info(f"solution: {solution}")

        n_questions = batch[0].shape[0] // num_rollouts

        self.infer.load()
        # Step 1: generate responses and masks.
        start_time = time.time()
        logger.info("generation start at {}-------------------------------".format(
            time.strftime('%H:%M:%S', time.localtime(start_time))))

        max_decode_length = self.grpo_config.max_decode_length
        if self.infer.use_vllm == VllmMode.ORIGIN:
            results = []
            input_bs = n_questions // self.infer_dp
            max_decode_length = self.grpo_config.max_decode_length
            for idx in range(num_rollouts * num_generations):
                result = self.infer.generate(input_ids_numpy[idx * input_bs: (idx + 1) * input_bs, :],
                                             max_decode_length=max_decode_length)
                for res_idx in range(len(result)):
                    if len(results) == len(result):
                        results[res_idx] = np.concatenate((results[res_idx], result[res_idx]))
                    else:
                        results.append(result[res_idx])
        else:

            results = self.infer.generate(input_ids_numpy, max_decode_length)

        end_time = time.time()
        print_perf_stat(start_time, end_time, "infer generate")

        logger.info("generation end at {}-------------------------------".format(
            time.strftime('%H:%M:%S', time.localtime(start_time))))

        self.infer.offload()
        logger.info("model_infer offload")

        logger.info(f"generate sequence results is {results} type {type(results)}")

        right_padding_responses, responses_mask_gather, \
            left_padding_prompts, prompts_mask_gather = self.infer.post_process_infer_outputs(results)
        all_prompts_mask = np.concatenate((prompts_mask_gather,
                                           np.zeros_like(responses_mask_gather, dtype=np.int32)), axis=1)
        all_responses_mask = np.concatenate(
            (np.zeros_like(left_padding_prompts, dtype=np.int32), responses_mask_gather),
            axis=1)
        all_prompt_completion_ids = np.concatenate((left_padding_prompts, right_padding_responses),
                                                   axis=1)
        # Step 3: calculate reward.
        start_time = time.time()
        logger.info("calculate reward start at {}-------------------------------".format(
            time.strftime('%H:%M:%S', time.localtime(start_time))))

        logger.info(
            f"left_padding_prompts is {type(left_padding_prompts)}")
        no_padding_prompts = self._remove_left_padding(left_padding_prompts,
                                                       padding_token=self.grpo_config.pad_token_id)
        no_padding_responses = self._remove_right_padding(
            right_padding_responses, padding_token=self.grpo_config.pad_token_id)
        prompts = self.tokenizer.decode(no_padding_prompts, skip_special_tokens=True)
        completions = self.tokenizer.decode(no_padding_responses, skip_special_tokens=True)

        logger.info(f"prompts: \n {prompts}")
        logger.info(f"completions: \n {completions}")

        all_elements_completion_len.extend([len(com) for com in completions])
        completions_length = np.array([len(com) for com in completions])
        mean_len = completions_length.mean()
        logger.info(f"mean completions.length: \n {mean_len}")

        rewards_per_func = np.zeros((n_questions * num_generations * num_rollouts, len(self.reward_funcs)),
                                    dtype=np.float32)
        answer_parsed_lst = []
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            if reward_func is accuracy_reward:
                output_reward_func, answer_parsed_lst = reward_func(prompts=prompts, completions=completions,
                                                                    **reward_kwargs)
            else:
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = np.array(output_reward_func, dtype=np.float32)
        rewards = (rewards_per_func * self.reward_weights[np.newaxis, :]).sum(axis=1)
        logger.info(f"precision rewards are {rewards}")
        logger.info(f"precision parse answer are {answer_parsed_lst}")

        end_time = time.time()
        print_perf_stat(start_time, end_time, "calculate reward")
        logger.info("calculate reward end at {}-------------------------------".format(
            time.strftime('%H:%M:%S', time.localtime(end_time))))

        all_rewards = np.array(rewards, dtype=np.float32)
        logger.info(f"loaded_all_rewards: {all_rewards}")

        total_size = all_rewards.shape[0]
        advantages = np.zeros((total_size,))
        tmp_all_rewards = all_rewards.copy()
        samples_per_step = total_size // num_generations
        for i in range(samples_per_step):
            listnum = list(range(i, total_size, samples_per_step))
            temp_rewards = tmp_all_rewards[listnum]
            adv_tem, mean_grouped_rewards = self.compute_advantages(temp_rewards)
            all_mean_grouped_rewards.append(mean_grouped_rewards)
            advantages[i::samples_per_step] = adv_tem.reshape((-1,))

        logger.info(f"advantages: {advantages}")
        logger.info(f"all_mean_grouped_rewards: {all_mean_grouped_rewards}")

        self.ref.load()
        logger.info("ref_model load")

        # regroup the index of all data
        # assume the input shape is [num_generations*num_rollout*num_questions, -1]
        # [1,2,3,4,1,2,3,4]--->[1,1,2,2,3,3,4,4]
        def reconstruct_index(x, num_generations):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=1)
            seq = x.shape[-1]
            x = x.reshape((num_generations, -1, seq)).transpose((1, 0, 2))
            return x.reshape((-1, seq))

        all_prompt_completion_ids = reconstruct_index(all_prompt_completion_ids, num_generations)
        all_prompts_mask = reconstruct_index(all_prompts_mask, num_generations)
        all_responses_mask = reconstruct_index(all_responses_mask, num_generations)
        advantages = reconstruct_index(advantages, num_generations)
        if not self.grpo_config.packing:
            pack_num = 1
        else:
            pack_num = self.grpo_config.pack_num
        all_packed = self.pack_grpo_data(
            all_prompt_completion_ids, all_prompts_mask, all_responses_mask, advantages, pack_num)
        total_ref_batch_size = self.grpo_config.ref_model_batch_size * self.ref_dp
        logger.info(
            f"total_ref_batch_size: ref_model_batch_size * ref_dp, {self.grpo_config.ref_model_batch_size} * "
            f"{self.ref_dp} = {total_ref_batch_size}")
        while len(all_packed) < total_ref_batch_size:
            all_packed.append(all_packed[0])

        all_ref_per_token_logps = np.zeros(
            (len(all_packed), self.grpo_config.seq_length), dtype=np.float32)
        ref_step_num = len(all_packed) // total_ref_batch_size
        logger.info(f"ref model total steps: {ref_step_num}")
        for idx in range(ref_step_num):
            # responses_mask will be updated before ref model infer.
            prompt_completion_ids, actual_sequence_length = self._construct_inputs_packing(
                all_packed, ref_model_batch_size=total_ref_batch_size, idx=idx)

            prompt_completion_ids = np.pad(prompt_completion_ids, ((0, 0), (0, 1)), 'constant',
                                           constant_values=self.grpo_config.pad_token_id)
            sampels_tensor = Tensor(prompt_completion_ids[:, 1:], dtype=ms.int32)
            input_prompt_ids = Tensor(prompt_completion_ids[:, :-1], dtype=ms.int32)
            actual_sequence_length = Tensor(actual_sequence_length, dtype=ms.int32)

            # Step 2: run ref model.
            start_time = time.time()
            logger.info("reference model step {} start at {}-------------------------------".format(
                idx, time.strftime('%H:%M:%S', time.localtime(start_time))))

            ref_per_token_logps = self.ref.compute_ref_log_prob(
                input_prompt_ids, None, samples=sampels_tensor, actual_sequence_length=actual_sequence_length)
            ref_per_token_logps = ref_per_token_logps.asnumpy().astype(np.float32)

            end_time = time.time()
            logger.info("reference model step {} end at {}, elapsed time {}-------------------------------".format(
                idx, time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time))
            all_ref_per_token_logps[
                idx * total_ref_batch_size: (idx + 1) * total_ref_batch_size, :] = ref_per_token_logps

        self.ref.offload()
        logger.info("ref_model offload")

        if (self.grpo_config.save_prompt_completions_data and
                step % self.grpo_config.save_prompt_completions_interval == 0):
            save_kwargs = {
                MetricData.QUESTION.value: prompts,
                MetricData.ANSWER.value: completions,
                MetricData.PARSED_ANSWER.value: answer_parsed_lst,
                MetricData.SOLUTION.value: solution,
                MetricData.REWARD_PER_QUESTION.value: rewards,
                MetricData.COMPLETION_LENGTH_PER_QUESTION.value: completions_length
            }
            save_file_path = os.path.join(self.grpo_config.save_prompt_completions_dir,
                                          f'prompt_completions_step_{step}.json')
            save_prompt_completions_data(save_file_path, **save_kwargs)

        for i in range(len(all_packed)):
            prompt_completion_ids_temp = np.pad(all_packed[i]["prompt_completion_ids"], ((0, 1),), 'constant',
                                                constant_values=self.grpo_config.pad_token_id).astype(np.int32)

            responses_mask_temp = np.pad(all_packed[i]["responses_mask"], ((0, 1),), 'constant',
                                         constant_values=0).astype(np.int32)
            responses_mask_temp = responses_mask_temp[1:]

            ref_per_token_logps = all_ref_per_token_logps[i].astype(np.float32)
            ref_per_token_logps = ref_per_token_logps * responses_mask_temp
            ref_per_token_logps[np.isnan(ref_per_token_logps)] = 0.0

            grpodata = GRPOData(
                prompt_completion_ids=prompt_completion_ids_temp.astype(np.int32),
                responses_mask=responses_mask_temp.astype(np.int32),
                ref_per_token_logps=ref_per_token_logps.astype(np.float32),
                advantages=all_packed[i]["advantages"].astype(np.float32),
                actual_sequence_length=all_packed[i]["actual_sequence_length"].astype(np.int32),
                sample_index=all_packed[i]["sample_index"].astype(np.int32),
                sample_valid_length=all_packed[i]["sample_valid_length"].astype(np.int32),
            )
            grpo_rl_elements.append(grpodata)

        logger.info(f"grpo_rl_elements.length: {len(grpo_rl_elements)}")
        all_mean_len = np.mean(all_elements_completion_len)
        logger.info(f"all_elements_completion_len mean: {all_mean_len}")
        if self.tensor_writer:
            self.tensor_writer.add_scalar("mean-completion-length", all_mean_len, global_step=self.make_exp_step)
        self.train.push_to_store(grpo_rl_elements)
        logger.info(f"all_mean_grouped_rewards: {all_mean_grouped_rewards}")
        avg_scores = np.mean(np.array(all_mean_grouped_rewards))
        logger.info(f"Avg scores: {avg_scores}")
        if self.tensor_writer:
            self.tensor_writer.add_scalar("average-scores", avg_scores, global_step=self.make_exp_step)

        end_time = time.time()
        print_perf_stat(ep_begin_time, end_time, "Make experience")
        logger.info("Make experience, end at {} ------------------------------- ".format(
            time.strftime('%H:%M:%S', time.localtime(end_time))))
        if self.grpo_config.save_data_file:
            if get_rank() % 8 == 0:
                self._save_grpoelement(self.grpo_config.save_data_file)
        self.make_exp_step += 1
        logger.info('generate over')

    def _print_data_str(self, data, name):
        decoded_str = self.tokenizer.decode(data)
        logger.info(f"{name} str is {decoded_str}")

    def _save_grpoelement(self, save_path):
        """ save grpo element """
        if save_path:
            logger.info(f"start save grpo {save_path}")
            schema = {
                "prompt_completion_ids": {"type": "int32", "shape": [-1]},
                "responses_mask": {"type": "int32", "shape": [-1]},
                "ref_per_token_logps": {"type": "float32", "shape": [-1]},
                "advantages": {"type": "float32", "shape": [-1]},
                "actual_sequence_length": {"type": "int32", "shape": [-1]},
                "sample_index": {"type": "int32", "shape": [-1]},
                "sample_valid_length": {"type": "int32", "shape": [-1]},
            }

            writer = FileWriter(file_name=save_path, shard_num=1, overwrite=True)
            writer.add_schema(schema)
            count = 0
            for ele in self.train.store:
                count += 1
                logger.info(f"add data element {asdict(ele)}")
                writer.write_raw_data([asdict(ele)])
            writer.commit()
            logger.info(f"end save grpo {save_path}")

    def run_grpo_train(self):
        """
        Main entry of MindRLHF GRPO training.
        """
        logger.info(
            f"Start training epoch num:{self.grpo_config.epochs}, step num:{self.step_num}, "
            f"generation num:{self.grpo_config.num_generations}")
        np.set_printoptions(threshold=1024)
        # 第一次执行前, load ckpt后参数在host上, 在网络第一次执行时会将参数自动加载到device上, 不需要手动load/offload
        for n in range(self.grpo_config.epochs):
            for i in range(self.step_num):

                step_begin_time = time.time()
                logger.info("step begin at {} \n------------------------------- "
                            .format(time.strftime('%H:%M:%S', time.localtime(step_begin_time))))

                logger.info(f"epoch: {n}, step: {i}")
                total_step = n * self.step_num + i
                self._make_experience(num_rollouts=self.grpo_config.num_rollouts,
                                      num_generations=self.grpo_config.num_generations,
                                      step=total_step)
                self.train.load_optimizer()
                self.train.load_model()

                self.train.train()
                self.train.offload_optimizer()

                # load for reshard
                if self.reshard_mem_opt_level == 1:
                    self.train.offload_model()
                    assert not self.train.model_on_device, ("when reshard_mem_opt_level is equal to 1, "
                                                            "train model must not on device before transform param")
                    assert not self.infer.on_device, ("when reshard_mem_opt_level is equal to 1, "
                                                      "infer model must not on device before transform param")
                else:
                    self.infer.load()
                    assert self.train.model_on_device, ("when reshard_mem_opt_level is equal to 0, "
                                                        "train model must not on device before transform param")
                    assert self.infer.on_device, ("when reshard_mem_opt_level is equal to 0, "
                                                  "infer model must not on device before transform param")

                if self.transform.sync_ref_model and \
                        ((i + 1) % self.transform.ref_model_sync_steps == 0):
                    # in some work, ref update may have a 'bad' effect
                    if self.reshard_mem_opt_level == 0:
                        self.ref.load()
                    input_on_device_flag_dict = {"policy2infer": (self.train.model_on_device, self.infer.on_device),
                                                 "policy2ref": (self.train.model_on_device, self.ref.on_device)}
                    self.transform.reshard_params(i, input_on_device_flag_dict)
                    if self.reshard_mem_opt_level == 0:
                        self.ref.offload()
                else:
                    input_on_device_flag_dict = {"policy2infer": (self.train.model_on_device, self.infer.on_device),
                                                 "policy2ref": (self.train.model_on_device, self.ref.on_device)}
                    self.transform.reshard_params(i, input_on_device_flag_dict)
                if self.reshard_mem_opt_level == 0:
                    assert self.train.model_on_device, ("when reshard_mem_opt_level is equal to 0, "
                                                        "train model must on device after transform param")
                    assert self.infer.on_device, ("when reshard_mem_opt_level is equal to 0, "
                                                  "infer model must on device after transform param")
                    self.train.offload_model()
                else:
                    assert not self.train.model_on_device, ("when reshard_mem_opt_level is equal to 1, "
                                                            "train model must not on device after transform param")
                    assert not self.infer.on_device, ("when reshard_mem_opt_level is equal to 1, "
                                                      "infer model must not on device after transform param")

                step_end_time = time.time()
                print_perf_stat(step_begin_time, step_end_time, f"epoch {n} step {i}")
                logger.info("step end at  {}\n------------------------------- ".format(
                    time.strftime('%H:%M:%S', time.localtime(step_end_time))))

        # save checkpoint
        self.train.load_model()
        self.train.save_checkpoint(
            rank_id=get_rank(), steps=self.grpo_config.epochs)
        logger.info("run grpo train end")

    def rename_safetensors_weights(self, args):
        """ rename safetensors and write output to param_name_map.json"""
        # 默认3个模型要加载的safetensors文件相同，用同一个config对象处理
        config = MindFormerConfig(args.sft_path_train)
        config.load_checkpoint = args.load_sft_checkpoint_train

        if config.model.model_config.get("qkv_concat", False):
            raise ValueError("safetensors only support qkv_concat=False for now")

        if get_rank() == 0:
            convert_func_lst = []
            convert_func_lst.append(self.infer.convert_map_dict)
            convert_func_lst.append(self.ref.convert_map_dict)
            convert_func_lst.append(self.train.convert_map_dict)
            convert_index_json_total(config.load_checkpoint,
                                     config.load_checkpoint, convert_func_lst, False)
        else:
            # wait for rank 0 to finish
            time.sleep(10)
        ms.mint.distributed.barrier()
        _pynative_executor.sync()
