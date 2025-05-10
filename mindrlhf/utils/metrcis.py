# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Model training metrics."""
from enum import Enum


class MetricData(Enum):
    '''Enum Metrics'''
    QUESTION = "question"
    ANSWER = "answer"
    PARSED_ANSWER = 'parsed_answer'
    SOLUTION = 'solution'
    REWARD_PER_QUESTION = 'reward'
    COMPLETION_LENGTH_PER_QUESTION = 'completion_length'

    REWARD_MEAN = 'reward_mean'
    REWARD_MIN = 'reward_min'
    REWARD_MAX = 'reward_max'

    ADVANTAGE_MEAN = 'advantage_mean'
    ADVANTAGE_MIN = 'advantage_min'
    ADVANTAGE_MAX = 'advantage_max'

    RESPONSE_LENGTH = 'response_length'
    RESPONSE_MIN = 'response_min'
    RESPONSE_MAX = 'response_max'
    RESPONSE_CLIP_RATIO = 'response_clip_ratio'
