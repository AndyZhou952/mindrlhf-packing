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

export GLOG_v=3

master_port=$1
case_name=$2

WORKDIR="$(realpath "$(dirname "$0")")"
echo "WORKDIR is $WORKDIR"
cd $WORKDIR
export MINDRLHF_PATH=$WORKDIR/../../
export MINDFORMERS_PATH=$WORKDIR/mindformers/
export PYTHONPATH=$MINDRLHF_PATH:$MINDFORMERS_PATH:$PYTHONPATH
echo "PYTHONPATH is $PYTHONPATH"
echo "case name is $case_name"

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
      --master_port=${master_port} --join=True --log_dir=./${case_name} \
      pytest -s transform_param_d2d.py::${case_name}
