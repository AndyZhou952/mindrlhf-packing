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
"""reshard optimizer test case"""

import pytest

from mindrlhf.utils.reshard_optimizer import ReshardOptimizer, Parallel, Layout


@pytest.mark.reshard_optimizer_basic
def test_same_1():
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=2, tp=4, pp=1), dst_parallel=Parallel(dp=2, tp=4, pp=1)
    )
    src_layout = Layout(dev_mat=[2, 4], tensor_map=[-1, 0])
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(dev_mat=[2, 4], tensor_map=[-1, 0])


@pytest.mark.reshard_optimizer_basic
def test_same_2():
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=4, tp=8, pp=1), dst_parallel=Parallel(dp=4, tp=8, pp=1)
    )
    src_layout = Layout(dev_mat=[4, 8], tensor_map=[0, -1])
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(dev_mat=[4, 8], tensor_map=[0, -1])


@pytest.mark.reshard_optimizer_basic
@pytest.mark.parametrize(
    "src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map",
    [
        ([2, 4], [-1, 0], [2, 2, 2], [-1, 1]),
        ([4, 2], [0], [2, 2, 2], [1]),
        ([2, 4], [0, -1], [2, 2, 2], [1, -1]),
        ([4, 2, 1], [1, 0], [2, 2, 2], [1, -1]),
        ([2, 4], [0], [2, 2, 2], [1]),
        ([4, 1, 2], [1, 0], [2, 2, 2], [-1, 1]),
    ],
)
@pytest.mark.reshard_optimizer_basic
def test_tp_1(src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map):
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=2, tp=4, pp=1), dst_parallel=Parallel(dp=4, tp=2, pp=1)
    )

    src_layout = Layout(dev_mat=src_dev_mat, tensor_map=src_tensor_map)
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(
        dev_mat=expected_dev_mat, tensor_map=expected_tensor_map
    )


@pytest.mark.reshard_optimizer_basic
@pytest.mark.parametrize(
    "src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map",
    [
        ([8, 128], [-1, 0], [8, 16, 8], [-1, 1]),
    ],
)
def test_tp_2(src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map):
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=8, tp=128, pp=1),
        dst_parallel=Parallel(dp=64, tp=16, pp=1),
    )

    src_layout = Layout(dev_mat=src_dev_mat, tensor_map=src_tensor_map)
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(
        dev_mat=expected_dev_mat, tensor_map=expected_tensor_map
    )


@pytest.mark.reshard_optimizer_basic
@pytest.mark.parametrize(
    "src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map",
    [
        ([4], [-1, 0], [2, 4], [-1, 0]),
        ([4], [0, -1], [2, 4], [0, -1]),
        ([4], [0], [2, 4], [0]),
    ],
)
def test_pp(src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map):
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=1, tp=4, pp=2), dst_parallel=Parallel(dp=2, tp=4, pp=1)
    )

    src_layout = Layout(dev_mat=src_dev_mat, tensor_map=src_tensor_map)
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(
        dev_mat=expected_dev_mat, tensor_map=expected_tensor_map
    )


@pytest.mark.reshard_optimizer_basic
@pytest.mark.parametrize(
    "src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map",
    [
        ([2, 4], [-1, 0], [2, 2, 2], [-1, 1]),
        ([2, 4], [0, -1], [2, 2, 2], [1, -1]),
        ([2, 4], [0], [2, 2, 2], [1]),
    ],
)
def test_tp_pp_1(src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map):
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=1, tp=4, pp=2), dst_parallel=Parallel(dp=4, tp=2, pp=1)
    )

    src_layout = Layout(dev_mat=src_dev_mat, tensor_map=src_tensor_map)
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(
        dev_mat=expected_dev_mat, tensor_map=expected_tensor_map
    )


@pytest.mark.reshard_optimizer_basic
@pytest.mark.parametrize(
    "src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map",
    [
        ([8], [0, -1], [4, 8], [0, -1]),
        ([8], [0], [4, 8], [0]),
        ([8], [-1, 0], [4, 8], [-1, 0]),
        ([8], [-1], [32, 1], [0]),
    ],
)
def test_tp_pp_2(src_dev_mat, src_tensor_map, expected_dev_mat, expected_tensor_map):
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(dp=1, tp=8, pp=4), dst_parallel=Parallel(dp=4, tp=8, pp=1)
    )

    src_layout = Layout(dev_mat=src_dev_mat, tensor_map=src_tensor_map)
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(
        dev_mat=expected_dev_mat, tensor_map=expected_tensor_map
    )


@pytest.mark.reshard_optimizer_basic
def test_tp_zero():
    """
    Feature: transform param with zero redundancy
    Description: dptppp transform
    Expectation: Run success
    """
    reshard_optimizer = ReshardOptimizer(
        src_parallel=Parallel(
            dp=2,
            tp=4,
            pp=1,
            enable_parallel_optimizer=True,
            parallel_optimizer_config={
                "parallel_optimizer_threshold": 0,
                "optimizer_weight_shard_size": 2,
            },
        ),
        dst_parallel=Parallel(dp=4, tp=2, pp=1),
    )
    src_layout = Layout(dev_mat=[2, 4], tensor_map=[-1, 0])
    dst_layout = reshard_optimizer.get_dst_layout(src_layout)
    assert dst_layout == Layout(dev_mat=[2, 2, 2], tensor_map=[-1, 1])
