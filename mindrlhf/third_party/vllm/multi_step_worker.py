# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/multi_step_worker.py

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.worker.multi_step_model_runner import (MultiStepModelRunner,
                                                 StatefulModelInput)
from vllm.worker.worker import WorkerInput
from vllm.worker.multi_step_worker import MultiStepState

from .worker import Worker


class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model_runner = self.model_runner
        # for multi-step model, wrap the model runner with MultiStepModelRunner
        self.model_runner = MultiStepModelRunner(
            base_model_runner,
            vllm_config=base_model_runner.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=base_model_runner.is_driver_worker,
        )

        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[
            Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None

    def _prepare_last_sampled_token_ids_for_tp_workers(
        self,
        execute_model_req: ExecuteModelRequest,
        model_input: StatefulModelInput,
    ) -> None:
        """ 
        Prepare the last sampled token ids for TP workers. If it's the last 
        PP rank, then the last sampled token ids are already in the model_input.
        If it is NOT the last PP rank, then we need to get the last sampled
        token that is cached in the execute_model_req.
        """
        if get_pp_group().is_last_rank:
            assert model_input.cached_outputs[
                -1].sampler_output.sampled_token_ids is None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None
            model_input.last_sampled_token_ids = model_input.cached_outputs[
                -1].sampled_token_ids
            # free sampled token ids from the previous step if it has been
            # pythonized. Cannot free the last sampled token ids because
            # we need it for GPU advance_step.
            for output in model_input.cached_outputs[:-1]:
                if output.pythonized:
                    output.sampled_token_ids = None
        else:
            # otherwise we need to get the cached sampled token ids from the
            # execute_model_req
            assert execute_model_req.last_sampled_token_ids is not None
            model_input.last_sampled_token_ids = (
                execute_model_req.last_sampled_token_ids.cuda())
            model_input.add_sampler_output(
                SamplerOutput(outputs=[], sampled_token_ids=None),
                model_input.last_sampled_token_ids)

            # free sampled token ids from the previous step.
            # TODO(will) we could reuse the sampled token ids tensor from
            # the previous step instead.
            for output in model_input.cached_outputs[:-1]:
                output.sampled_token_ids = None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None
