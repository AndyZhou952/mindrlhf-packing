# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed
import torch.nn as nn

# TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
from vllm.distributed import init_distributed_environment, set_custom_all_reduce
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.model_runner_base import ModelRunnerInputBase
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
from vllm.worker.multi_step_model_runner import (MultiStepModelRunner,
                                                 StatefulModelInput)
from vllm.worker.worker_base import WorkerBase, WorkerInput
from vllm.config import VllmConfig
from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.worker.multi_step_worker import MultiStepState

from .config import LoadFormat
from .hf_weight_loader import load_hf_weights
from .megatron_weight_loaders import load_megatron_weights
from .model_runner import ModelRunner
from .parallel_state import ensure_model_parallel_initialized

logger = init_logger(__name__)


class Worker(Worker):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
        ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.num_scheduler_steps = vllm_config.scheduler_config.num_scheduler_steps
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}

        # TODO(sgm): set correct model runner class
        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine] = None
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

        # NOTE(sgm): [VERL] For offloading inference engine params
        self.cpu_model = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
            self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{local_rank}")
            if self.rank < 0:
                raise ValueError("Invalid or unspecified rank.")
            # TODO(tronzhang): Upper init, skip this device set.
            # torch.cuda.set_device(self.device)

            # Use the world_size set by TORCHRUN
            world_size = int(os.getenv("WORLD_SIZE", "-1"))
            assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
            self.parallel_config.world_size = world_size

            # TODO(tronzhang): skip check here
            # _check_if_gpu_supports_dtype(self.model_config.dtype)
            # gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        # self.model = get_model(actor_model=self.model, model_config=self.model_config)

    def _init_cache_engine(self):
        if self.cache_engine is None and self.gpu_cache is None:
            super()._init_cache_engine()
            if self.num_scheduler_steps > 1:
                self.model_runner._base_model_runner.model.mf_kvcaches_init = False # mindformers model
            else:
                self.model_runner.model.mf_kvcaches_init = False

    def free_cache_engine(self):
        # ensure `enforce_eager=True`
        del self.cache_engine
        del self.gpu_cache
        gc.collect()
        self.cache_engine = None
        self.gpu_cache = None

    def execute_model(self,
                      execute_model_req: ExecuteModelRequest,
                      intermediate_tensors: Optional[IntermediateTensors] = None) -> Optional[List[SamplerOutput]]:
        if self.num_scheduler_steps > 1:
            return self.execute_model_mss(execute_model_req, intermediate_tensors)
        else:
            return self.execute_model_without_mss(execute_model_req, intermediate_tensors)

    # NOTE(sgm): [VERL]: adapt from _execute_model_spmd()
    def execute_model_without_mss(self,
                                  execute_model_req: ExecuteModelRequest,
                                  intermediate_tensors: Optional[IntermediateTensors] = None) -> Optional[List[SamplerOutput]]:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, ("_execute_model_spmd() requires each worker to take in an "
        "ExecuteModelRequest")
        worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = self.model_runner.prepare_model_input(
        execute_model_req.seq_group_metadata_list)

        # verl.worker.workerbase.WorkerBase
        # swap cache
        super().execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        return self.model_runner.execute_model(
            model_input,
            self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
            intermediate_tensors,
        )

    # NOTE(sgm): [VERL]: adapt from _execute_model_spmd()
    def execute_model_mss(self,
                          execute_model_req: ExecuteModelRequest,
                          intermediate_tensors: Optional[IntermediateTensors] = None) -> Optional[List[SamplerOutput]]:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, ("_execute_model_spmd() requires each worker to take in an "
                                               "ExecuteModelRequest")
        is_first_multi_step = execute_model_req.is_first_multi_step
        virtual_engine = execute_model_req.virtual_engine
        if is_first_multi_step:
            # on first step we prepare the worker input and model input normally
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            model_input: StatefulModelInput = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))
            if execute_model_req.async_callback:
                model_input.frozen_model_input = dataclasses.replace(  # type: ignore
                    model_input.frozen_model_input,
                    async_callback=execute_model_req.async_callback)
        else:
            # on subsequent steps we reuse the worker input and model input
            multi_step_state = self.multi_step_states[virtual_engine]
            worker_input = multi_step_state.worker_input
            model_input = multi_step_state.model_input
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
            assert frozen_model_input.attn_metadata is not None
            # clear the cached metadata so that it can be recomputed on
            # the workers.
            frozen_model_input.attn_metadata._cached_prefill_metadata = None
            frozen_model_input.attn_metadata._cached_decode_metadata = None

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step

        if not is_first_multi_step:
            # we broadcast the last sampled token ids to all TP workers so they
            # can update their model input metadata in-place.
            self._prepare_last_sampled_token_ids_for_tp_workers(
                execute_model_req=execute_model_req, model_input=model_input)

        if execute_model_req.is_first_multi_step:
            # cache the worker input and model input for the next steps
            self.multi_step_states[virtual_engine] = MultiStepState(
                worker_input=worker_input, model_input=model_input)

        # verl.worker.workerbase.WorkerBase
        # swap cache
        super().execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        return self.model_runner.execute_model(
            model_input,
            self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
            intermediate_tensors,
        )
    
    def get_model(self):
        if self.num_scheduler_steps > 1:
            return self.model_runner._base_model_runner.model.network # mindformers model
        else:
            return self.model_runner.model.network # mindformers model


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = "env://",
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # NOTE(sgm) use tcp://localhost:xxxx will hang in HF setting without megatron
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank)

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
    )


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
