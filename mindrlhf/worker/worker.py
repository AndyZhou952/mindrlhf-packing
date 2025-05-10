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
"""worker"""

from abc import ABC
from dataclasses import dataclass
import numpy as np


@dataclass
class GRPOData:
    """
    grpo data
    """

    prompt_completion_ids: np.array
    responses_mask: np.array
    ref_per_token_logps: np.array
    advantages: np.array
    actual_sequence_length: np.array
    sample_index: np.array
    sample_valid_length: np.array


# 占位，后续扩展基类功能
class Worker(ABC):
    """
    Args:
       paradigm (str): programming paradigm, can be set to 'SPMD' or 'MPMD'.
                       For 'MPMD' paradigm, this worker will be ray.remote to resource pool.
    """

    def __init__(self, paradigm="SPMD", mpmd_args=None):
        pass


def format_time_delta(seconds):
    """format time delta to string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:.4f}"
