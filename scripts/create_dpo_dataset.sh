#!/bin/bash
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

# Generates a dummy dataset in /path/to/train_dpo_format.jsonl

NUM_EXAMPLES_TO_GENERATE=$1
data_path=$2

for i in $(seq 1 $NUM_EXAMPLES_TO_GENERATE); do
   cat <<EOF
{"prompt": "<extra_id_0>System\n\n<extra_id_1>User\n${i}*10=?\n<extra_id_1>Assistant\n", "pos_resp": "$((i * 10))\n<extra_id_1>", "neg_resp": "I refuse to answer this question.\n<extra_id_1>", "pos_type": "拒绝为主", "neg_type": "风险回复"}
EOF
done | tee $data_path >/dev/null