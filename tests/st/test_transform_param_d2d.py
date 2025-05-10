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
"""transform param from device to device test case"""
import os
import pytest

root_path = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_transform_d2d_no_pp_1():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10801 test_transform_d2d_no_pp_1"
    )
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_transform_d2d_no_pp_2():
    """
    Feature: transform param no pp scenario
    Description: dpmp from zero to no zero transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10802 test_transform_d2d_no_pp_2"
    )
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_transform_d2d_no_pp_3():
    """
    Feature: transform param no pp scenario
    Description: dpmp from zero to zero transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10803 test_transform_d2d_no_pp_3"
    )
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_transform_d2d_no_pp_4():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10804 test_transform_d2d_no_pp_4"
    )
    assert return_code == 0


def test_transform_d2d_pp_1():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10805 test_transform_d2d_pp_1"
    )
    assert return_code == 0


def test_transform_d2d_pp_2():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10806 test_transform_d2d_pp_2"
    )
    assert return_code == 0


def test_transform_d2d_pp_3():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10807 test_transform_d2d_pp_3"
    )
    assert return_code == 0


def test_transform_d2d_pp_4():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10808 test_transform_d2d_pp_4"
    )
    assert return_code == 0


def test_transform_d2d_with_reshard_optimizer_tp():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10809 test_transform_d2d_with_reshard_optimizer_tp"
    )
    assert return_code == 0


def test_transform_d2d_with_reshard_optimizer_tp_zero():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        f"bash {root_path}/run_transform_param_test.sh 10810 test_transform_d2d_with_reshard_optimizer_tp_zero"
    )
    assert return_code == 0
