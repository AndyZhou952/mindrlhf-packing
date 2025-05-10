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

import subprocess


def check_log(file_path, check_pairs=None):
    # check the number of key in check_pairs in log file is equal to the value
    if check_pairs is not None:
        for key_word, value in check_pairs.items():
            log_output = subprocess.check_output(
                ["grep -r '%s' %s | wc -l" % (key_word, file_path)],
                shell=True)
            log_cnt = str(log_output, 'utf-8').strip()
            assert log_cnt == str(value), (f"Failed to find {key_word} in {file_path} or content is not correct."
                                           f"Expected occurrences: {value}, but got {log_cnt}")