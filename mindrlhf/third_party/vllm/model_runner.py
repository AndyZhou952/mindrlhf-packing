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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py
"""Model runner"""
import warnings
from enum import IntEnum
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import vllm.envs as envs
from vllm.config import CompilationLevel, VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.prompt_adapter.worker_manager import LRUCacheWorkerPromptAdapterManager
from vllm.utils import DeviceMemoryProfiler, supports_dynamo
from vllm.worker.model_runner import ModelRunner
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.platforms import current_platform

from .config import LoadConfig, ModelConfig
from .model_loader import get_model

logger = init_logger(__name__)


class ModelRunner(ModelRunner):
    """Model runner"""
    def __init__(
            self,
            vllm_config: VllmConfig,
            kv_cache_dtype: Optional[str] = "auto",
            is_driver_worker: bool = False,
            return_hidden_states: bool = False,
            input_registry: InputRegistry = INPUT_REGISTRY,
            mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        ):
        super().__init__(
            vllm_config,
            kv_cache_dtype,
            is_driver_worker=True,  # a hack
            return_hidden_states=return_hidden_states,
            input_registry=input_registry,
            mm_registry=mm_registry,
        )

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            import mindspore as ms
            self.model = get_model(vllm_config=self.vllm_config)
            self.model.network.set_dynamic_inputs()
            dynamic_hidden_states = ms.Tensor(shape=[None, None], dtype=self.model.mf_model_config.compute_dtype)
            self.model.network.lm_head.set_inputs(dynamic_hidden_states)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))
