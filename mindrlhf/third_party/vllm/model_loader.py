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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
"""Utilities for selecting and loading models."""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from vllm.config import CacheConfig, DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.loader import _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from .config import LoadConfig, LoadFormat, ModelConfig
from .hf_weight_loader import update_hf_weight_loader
from .megatron_weight_loaders import load_megatron_weights, update_megatron_weight_loader


def get_model(
    vllm_config: VllmConfig,
) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    return loader.load_model(vllm_config=vllm_config)


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.AUTO:
        update_megatron_weight_loader()
        return MegatronLoader(load_config)

    # NOTE(sgm): change the weight_loader function in runtime
    if load_config.load_format == LoadFormat.MEGATRON:
        update_megatron_weight_loader()
        return MegatronLoader(load_config)

    if load_config.load_format == LoadFormat.HF:
        update_hf_weight_loader()
        return HFLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_HF:
        update_hf_weight_loader()
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_MEGATRON:
        update_megatron_weight_loader()
        return DummyModelLoader(load_config)

    raise ValueError("load format not supported in verl: {}, only support {} and {}".format(
        load_config.load_format, LoadFormat.MEGATRON, LoadFormat.HF))


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self,
        *,
        vllm_config: VllmConfig
    ) -> nn.Module:
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with torch.device(vllm_config.device_config.device):
                model = _initialize_model(vllm_config=vllm_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            # initialize_dummy_weights(model)
        return model.eval()


class MegatronLoader(BaseModelLoader):
    """Model loader that can load the model weights from partitioned megatron model."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        pass

    def load_model(
        self,
        # actor_model: Union[PreTrainedModel, Dict],
        vllm_config: VllmConfig
    ) -> nn.Module:
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with torch.device(vllm_config.device_config.device):
                model = _initialize_model(vllm_config=vllm_config)

        return model.eval()


class HFLoader(BaseModelLoader):
    """Model loader that can load the model weights from model's full params."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def _get_weights_iterator(self, actor_model: Union[PreTrainedModel, Dict]):
        if isinstance(actor_model, Dict):
            return actor_model.items()
        elif isinstance(actor_model, nn.Module):
            return dict(actor_model.named_parameters()).items()
        else:
            raise ValueError(f"actor model should be Dict or nn.Module, but get {type(actor_model)}")

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        vllm_config: VllmConfig
    ) -> nn.Module:
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            # with torch.device(vllm_config.device_config.device):
            # NOTE(sgm): init the model in cpu
            model = _initialize_model(vllm_config=vllm_config)
            model.load_weights(self._get_weights_iterator(actor_model))
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        # model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        return model.eval()
