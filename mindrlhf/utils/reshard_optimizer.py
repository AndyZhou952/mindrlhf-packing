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
"""Reshard Optimizer"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import itertools
from mindformers import logger


OPT_COMMUNICATION_GROUPS = None


@dataclass
class Layout:
    """Layout"""

    dev_mat: List
    tensor_map: List


@dataclass
class Parallel:
    """Parallel"""

    dp: int = 1
    tp: int = 1
    pp: int = 1
    enable_parallel_optimizer: bool = False
    parallel_optimizer_config: Dict = None


class ReshardOptimizer:
    """Reshard Optimizer"""

    def __init__(self, src_parallel: Parallel, dst_parallel: Parallel):
        """
        Initialize the ReshardOptimizer.
        """
        self.src_parallel = src_parallel
        self.dst_parallel = dst_parallel

        self.world_size = self.dst_parallel.dp * self.dst_parallel.tp * self.dst_parallel.pp
        if self.world_size != self.src_parallel.dp * self.src_parallel.tp * self.src_parallel.pp:
            raise ValueError("The world size of src_parallel and dst_parallel should be the same")

        self.opt_communication_groups = self._get_opt_communication_groups()
        logger.info(
            f"Reshard Optimizer is created successfully! src_parallel: {src_parallel}, "
            f"dst_parallel: {dst_parallel}, opt_communication_groups: {self.opt_communication_groups}"
        )

    def get_dst_layout(self, layout: Layout) -> Layout:
        """
        Get the destination layout based on the source layout.
        """
        cut_axises = [
            index
            for index, value in enumerate(layout.tensor_map)
            if (value != -1 and layout.dev_mat[-(value + 1)] != 1)
        ]
        if len(cut_axises) > 1:
            raise ValueError(f"The axis of the layout is cut invalid: {layout}")

        if len(cut_axises) == 1:
            same_data_labels = self._get_same_data_labels(self.opt_communication_groups["tp"])

            dev_mat, index = self._find_layout(same_data_labels)
            if not dev_mat:
                raise ValueError(
                    f"Find layout failed, please check the communication group: {self.opt_communication_groups['tp']}"
                )

            tensor_map = [-1] * len(layout.tensor_map)
            tensor_map[cut_axises[0]] = len(dev_mat) - 1 - index
            return Layout(dev_mat, tensor_map)
        return Layout([self.world_size, 1], [0])

    @classmethod
    def _find_layout(cls, same_data_labels: List) -> Tuple[List, int]:
        """
        Finds a suitable device matrix layout and column index that matches the given same_data_labels.

        Args:
        same_data_labels (List): A list of integers representing the target column to match.

        Returns:
        Tuple[List, int]: A tuple containing the device matrix (list of integers) and the column index (int).
                         Returns (None, None) if no match is found.

        Example:
            >>> same_data_labels = [0, 0, 1, 1]
            >>> dev_matrix, col_idx = _find_layout(same_data_labels)
            >>> print(dev_matrix, col_idx)
            [2, 2], 0
        """
        n = len(same_data_labels)
        factorizations = cls._get_factorizations(n)
        for factorization in factorizations:
            if len(factorization) == 1:
                factorization.append(1)

        factorizations.sort(key=len)
        visited = set()

        for factors in factorizations:
            for perm in set(itertools.permutations(factors)):
                if perm in visited:
                    continue
                visited.add(perm)

                dev_matrix = list(perm)
                ans = cls._generate_ans_matrix(dev_matrix)

                ans_col = list(zip(*ans))

                for col_idx, col in enumerate(ans_col):
                    if list(col) == same_data_labels:
                        return dev_matrix, col_idx
        return None, None

    def _get_opt_communication_groups(self) -> Dict:
        """
        get optimal groups based on the source parallelism and destination parallelism.
        """
        group_ranks = []
        if self.src_parallel.tp % self.dst_parallel.tp != 0:
            raise ValueError("The source tp should be divisible by the destination tp")

        num_tensor_model_parallel_groups_per_train_tp = self.src_parallel.tp // self.dst_parallel.tp
        num_tensor_model_parallel_groups = self.world_size // self.dst_parallel.tp

        if num_tensor_model_parallel_groups_per_train_tp == 1:
            for i in range(num_tensor_model_parallel_groups):
                ranks = range(i * self.dst_parallel.tp, (i + 1) * self.dst_parallel.tp)
                group_ranks.append(list(ranks))
        else:
            train_tp = num_tensor_model_parallel_groups_per_train_tp * self.dst_parallel.tp

            for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
                start = train_tp * i
                end = train_tp * (i + 1)
                for j in range(num_tensor_model_parallel_groups_per_train_tp):
                    ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                    for k in range(len(ranks)):
                        ranks[k] += j
                    group_ranks.append(ranks)
        return {"tp": group_ranks, "dp": [list(group) for group in zip(*group_ranks)]}

    @staticmethod
    def _get_same_data_labels(tp_groups) -> List:
        """
        Generates a list where each element represents the position of a rank within its group.

        Args:
            tp_groups (List[List[int]]): A list of groups, where each group is a list of integers representing ranks.

        Returns:
            List[int]: A list where each index corresponds to a rank,
            and the value is the position of that rank within its group.

        Example:
            >>> tp_groups = [[0, 2], [1, 3], [4, 6], [5, 7]]
            >>> _get_same_data_labels(tp_groups)
            [0, 0, 1, 1, 0, 0, 1, 1]
        """
        all_ranks = [rank for group in tp_groups for rank in group]
        if not all_ranks:
            return []
        max_rank = max(all_ranks)
        output = [0] * (max_rank + 1)
        for group in tp_groups:
            for pos, rank in enumerate(group):
                output[rank] = pos
        return output

    @staticmethod
    def _get_factorizations(n):
        """
        Generates all possible factorizations of a positive integer n into factors greater than or equal to 2.

        This function uses a backtracking algorithm to recursively find all possible combinations of factors
        whose product equals n. Each factorization is returned as a list of integers in ascending order.

        Args:
            n (int): The positive integer to factorize.

        Returns:
            list[list[int]]: A list of all possible factorizations of n. Each factorization is a list of integers.

        Example:
            >>> _get_factorizations(12)
            [[2, 2, 3], [2, 6], [3, 4], [12]]
            >>> _get_factorizations(8)
            [[2, 2, 2], [2, 4], [8]]
        """
        result = []

        def backtrack(product, start, current_factors):
            if product == 1:
                result.append(current_factors[:])
                return

            for factor in range(start, product + 1):
                if product % factor == 0:
                    current_factors.append(factor)
                    backtrack(product // factor, factor, current_factors)
                    current_factors.pop()

        backtrack(n, 2, [])
        return result

    @classmethod
    def _generate_ans_matrix(cls, dev_matrix):
        """
        Generates a matrix where each row represents an integer converted to a list of digits
        in a mixed-radix numeral system.

        Args:
            dev_matrix (list[int]): A list of integers representing the bases for each digit position.

        Returns:
            list[list[int]]: A matrix where each row is the mixed-radix representation of an integer from 0 to N-1.

        Example:
            >>> dev_matrix = [3, 2]
            >>> _generate_ans_matrix(dev_matrix)
            [
                [0, 0],  # 0 in mixed-radix (3, 2)
                [0, 1],  # 1 in mixed-radix (3, 2)
                [1, 0],  # 2 in mixed-radix (3, 2)
                [1, 1],  # 3 in mixed-radix (3, 2)
                [2, 0],  # 4 in mixed-radix (3, 2)
                [2, 1]   # 5 in mixed-radix (3, 2)
            ]
        """

        def int_to_base_list(i, dev_matrix):
            digits = []
            for base in reversed(dev_matrix):
                digits.append(i % base)
                i //= base
            return list(reversed(digits))

        n = 1
        for b in dev_matrix:
            n *= b

        ans = []
        for i in range(n):
            ans.append(int_to_base_list(i, dev_matrix))
        return ans
