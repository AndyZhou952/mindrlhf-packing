#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
"""Ascend platform Patch."""

import vllm
from vllm.config import VllmConfig


def check_and_update_config(vllm_config: VllmConfig) -> None:
    """
    Check and update the configuration for the current platform.

    It can raise an exception if the configuration is not compatible with
    the current platform, or it can update the configuration to make it
    compatible with the current platform.

    The config is passed by reference, so it can be modified in place.
    """
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config

    if parallel_config.worker_cls == "auto":
        import vllm.envs as envs

        if scheduler_config.is_multi_step:
            # pylint: disable=R1720
            if envs.VLLM_USE_V1:
                raise NotImplementedError
            else:
                parallel_config.worker_cls = (
                    "vllm.worker.multi_step_worker.MultiStepWorker"
                )
        elif vllm_config.speculative_config:
            # pylint: disable=R1720
            if envs.VLLM_USE_V1:
                raise NotImplementedError
            else:
                parallel_config.worker_cls = (
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                )
            parallel_config.sd_worker_cls = "vllm.worker.worker.Worker"
        else:
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            else:
                parallel_config.worker_cls = "vllm.worker.worker.Worker"

    cache_config = vllm_config.cache_config
    if cache_config and cache_config.block_size is None:
        cache_config.block_size = 16


vllm.config.current_platform.check_and_update_config = check_and_update_config
vllm.platforms.current_platform.check_and_update_config = check_and_update_config
vllm.utils.current_platform.check_and_update_config = check_and_update_config
