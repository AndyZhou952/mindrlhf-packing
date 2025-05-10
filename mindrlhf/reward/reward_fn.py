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
"""Reward functions for GRPO training."""
import re

from .verifier.rule_verifier import func_from_jiaoda

# pylint: disable=W0613
def reward_func_from_jiaoda(completions, solution, **kwargs):
    return func_from_jiaoda(completions, solution)


# pylint: disable=W0613
def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    answer_parsed_lst = []

    for content, sol in zip(completions, solution):
        response = re.sub(r"(\d),(\d)", r"\1\2", content)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        if numbers:
            predictions = numbers[-1]
        else:
            predictions = response
        sol = str(re.findall(r'\d+', sol)[0])
        ground_truth_answer = re.sub(r"(\d),(\d)", r"\1\2", sol)
        reward = str(predictions).lower() == str(ground_truth_answer).lower()
        reward = 1.0 if reward else 0.0
        rewards.append(reward)
        predictions_len = len(str(predictions).lower())
        if predictions_len == 0:
            answer_parsed_lst.append('NO ANSWER')
        else:
            answer_parsed_lst.append(str(predictions).lower())

    return rewards, answer_parsed_lst


# pylint: disable=W0613
def accuracy_reward_2(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    latex2sympy2_extended = importlib.import_module("latex2sympy2_extended")
    # pylint: disable=C0103
    NormalizationConfig = latex2sympy2_extended.NormalizationConfig
    math_verify = importlib.import_module("math_verify")
    # pylint: disable=C0103
    LatexExtractionConfig = math_verify.LatexExtractionConfig
    parse = math_verify.parse
    verify = math_verify.verify

    rewards = []
    answer_parsed_lst = []
    for content, sol in zip(completions, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        # pylint: disable=C1801
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
            if len(answer_parsed) == 0:
                answer_parsed_lst.append('NO ANSWER')
            elif len(answer_parsed) == 1:
                answer_parsed_lst.append(answer_parsed[0])
            else:
                answer_parsed_lst.append(answer_parsed[1])
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            print("Failed to parse gold solution: ", sol)
            answer_parsed_lst.append('solution parse failed')
        rewards.append(reward)

    return rewards, answer_parsed_lst

# pylint: disable=W0613
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match else 0.0 for match in matches]
