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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py

import os
from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.engine.arg_utils import EngineArgs
from vllm.config import CompilationConfig, VllmConfig
from vllm.usage.usage_lib import UsageContext

from .config import LoadConfig, ModelConfig


@dataclass
class EngineArgs(EngineArgs):
    model_hf_config: PretrainedConfig = None  # for verl

    def __post_init__(self):
        # Override the default value of enable_prefix_caching if it's not set
        # by user.
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = bool(envs.VLLM_USE_V1)

        # Override max_num_seqs if it's not set by user.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256 if not envs.VLLM_USE_V1 else 1024

        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, (int, dict)):
            self.compilation_config = CompilationConfig.from_cli(
                str(self.compilation_config))

    def create_model_config(self) -> ModelConfig:
        return ModelConfig(
            hf_config=self.model_hf_config,
            task=self.task,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            enforce_eager=self.enforce_eager,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            use_async_output_proc=False, # not self.disable_async_output_proc TODO[wyc]
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            override_neuron_config=self.override_neuron_config,
            override_pooler_config=self.override_pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config)

    def create_load_config(self) -> LoadConfig:
        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )


    def create_engine_config(self,
                             usage_context: Optional[UsageContext] = None
                             ) -> VllmConfig:
        engine_config = super().create_engine_config()

        # NOTE[VERL]: Use the world_size set by torchrun
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
        engine_config.parallel_config.world_size = world_size

        return engine_config
