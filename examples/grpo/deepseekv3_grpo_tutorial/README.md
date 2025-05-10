# Qwen-GRPO强化学习训练教程

GRPO（Group Relative Policy Optimization）是针对数学逻辑推理任务提出的强化学习优化算法。训练过程是通过学习一个策略模型，通过不断试错，策略模型与奖励模型的不断交互，策略模型会逐渐倾向于选择获得更高奖励的行为，自主探索出最佳学习路径。通过GRPO算法大规模训练，大模型在逻辑推理能力上得到显著提升。

本教程基于`Deepseek`模型与`GSM8K Train`数据集引导读者跑通单机8卡GRPO训推一体的基本流程。

## 一、模型以及数据集获取与预处理

### 模型权文件和tokenizer获取

用户可以从[魔搭社区](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1/files)下载完整预训练权重，模型对应的tokenizer文件`tokenizer.json`也可在上述链接中下载。

模型权重下载完成后，需要转为MindSpore使用的.ckpt文件。首先进入[MindFormers](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/convert_weight.py)路径

```shell
cd /{path}/mindformers
```

并执行以下脚本：

```shell
python research/deepseek3/convert_weight.py --torch_ckpt_path TORCH_CKPT_DIR --mindspore_ckpt_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:            模型名称
torch_ckpt_path:  下载HuggingFace权重的文件夹路径
output_path:      转换后的MindSpore权重文件保存路径
dtype:            转换权重的精度
```

脚本会将完整的.ckpt格式模型权重保存在`{path}/MS_CKPT_NAME.ckpt`路径下。

### 模型权重离线切分

当前版本的MindRLHF尚不支持权重在线切分，在使用多卡分布式训练时，需要用户手动进行权重切分，可参考[mindformers](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/README.md)教程。
需要注意，在GRPO算法中存在训练和推理两份模型权重，若训练和推理所使用的分布式策略不同，则需要分别切分两份分布式权重。

### 数据集文件获取与预处理

用户可以从[GSM8K Github Repo](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/)下载得到
`GSM8K Train`数据集`train.jsonl`。下载完成后，需要转为MindSpore使用的.mindrecord文件。首先进入MindRLHF路径

```shell
cd /{path}/mindrlhf
```

并执行以下脚本：

```shell
python examples/grpo/deepseekv3_grpo_tutorial/rlhf_data.py \
--tokenizer_file /{path}/tokenizer.json \
--file_path /{path}/train.jsonl/ \
--output_path /{path}/gsm8k_train.mindrecord

# 参数说明
tokenizer_file:       deepseek模型对应的tokenizer文件tokenizer.json路径
file_path:        GSM8K Train数据集train.jsonl文件路径
output_path:      输出.mindrecord文件路径
```

其中`tokenizer.json`都可以从Huggingface社区或魔搭社区对应模型页面获取。
此脚本会将`train.jsonl`转换成mindrecord的形式保存在`/{path}/gsm8k_train.mindrecord`。此数据路径将在训练拉起时作为`mind_dataset_dir`的值被传入。

## 二、GRPO算法及模型配置

### 依赖版本

当前版本所依赖框架:

| 依赖 | 版本 |
|------|----|
| 固件&驱动 | 24.1.RC3.3 |
| CANN | 8.0 |
| Python | 3.10 |
| MindSpore | master, commit id：b6b6fcd90e566dc2821f88904eea746db8690081 |
| MindFormers | dev, commit id：129f4459b0fc971cfd473759c4a0453120fb58ca |

### 训练/推理模型配置

训练模型的配置文件默认为`model_configs/deepseek_v3_config/finetune_deepseek3_671b.yaml`,其中用户可以手动配置训练模型的并行策略：

```shell
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 2
  expert_parallel: 1
  micro_batch_num:  1
# 参数说明
data_parallel:                数据并行切分组数
model_parallel:               模型并行(tensor parallel)切分组数
pipeline_stage:               流水线并行切分组数
expert_parallel:              专家并行切分数
micro_batch_num:              流水线并行中的micro batch number
```

推理模型的配置文件默认为`model_configs/deepseek_v3_config/predict_deepseek3_671b.yaml`,其中用户可以手动配置推理模型的并行策略：

```shell
parallel_config:
    data_parallel: 1
    model_parallel: 4
    pipeline_stage: 1
# 参数说明
data_parallel:                数据并行切分组数
model_parallel:               模型并行(tensor parallel)切分组数
pipeline_stage:               流水线并行切分组数, 必须为1. 当前推理模型不支持流水线并行
```

### GRPO训练算法配置

GRPO训练算法相关配置可以在`mindrlhf/configs/grpo_configs.py`内进行修改，包括以下参数：

```shell
beta: float = 0.01
num_generations: int = 8
grpo_epochs: int = 2
start_lr: float = 5e-7
end_lr: float = 1e-10
chunk_size: int = 2
batch_size: int = 2
sync_ref_model: bool = True
ref_model_sync_steps: int = 50

# 参数说明
grpo_epochs:            在数据集上总共训练的epochs轮数
chunk_size:             推理模型在每一步中为多少个问题生成回答
num_generations:        推理模型在每一步中为每个问题生成多少个回答
beta:                   反向训练GRPO loss中KL散度的权重
start_lr:               初始时反向训练的learning rate步长
end_lr:                 结束时反向训练的learning rate步长, 必须大于start_lr
batch_size:             反向训练的batch size
sync_ref_model:         是否每隔若干步将ref model的权重更新为最新的训练模型权重
ref_model_sync_steps:   若sync_ref_model=True, ref model权重更新的间隔步数
```

## 三、启动单机8卡GRPO训练脚本

首先进入MindRLHF路径

```shell
cd /{path}/mindrlhf
```

使用以下命令将本地mindrlhf和mindformers代码库均加入PYTHONPATH，MINDFORMERS_PAT路径中

```shell
export GLOG_v=2
export PYTHONPATH=/path/to/mindrlhf/:$PYTHONPATH
export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH
# use for vllm
export vLLM_MODEL_BACKEND=MindFormers
export MINDFORMERS_MODEL_CONFIG="./model_configs/deepseek_v3_grpo/predict_deepseek3_671b.yaml"
# 性能优化
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_TOTAL_MEMORY_GB=64
export vLLM_MODEL_MEMORY_USE_GB=50
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
export MS_DEV_RUNTIME_CONF="multi_stream:true"
#ray组网相关
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
#通信下发
export HCCL_OP_EXPANSION_MODE=AIV
# 关闭多机lccl
export MS_ENABLE_LCCL=off

# 参数说明
vLLM_MODEL_BACKEND: 指定使用mindformers模型，使用vllm时开启
MINDFORMERS_MODEL_CONFIG： 指定推理yaml，使用vllm时开启
vLLM_MODEL_MEMORY_USE_GB：性能优化相关
ASCEND_TOTAL_MEMORY_GB： 内存管理相关
MS_DEV_RUNTIME_CONF：性能优化相关
MS_DEV_RUNTIME_CONF="multi_stream:true"：在通信并发下有带宽抢占引发的性能劣化, 控制通信单流来规避该劣化
MS_ALLOC_CONF：优化碎片内存
GLOO_SOCKET_IFNAME：ray组网通信相关
TP_SOCKET_IFNAME：rat组网通信相关
ENABLE_LAZY_INLINE_NO_PIPELINE： 优化模型编译性能
HCCL_OP_EXPANSION_MODE：通信下发优化
MS_ENABLE_LCCL: 关闭多机lccl
```

随后使用以下命令拉起单机4卡GRPO训练任务

```shell
bash scripts/msrun_launcher.sh "examples/grpo/deepseekv3_grpo_tutorial/main.py \
--config examples/grpo/deepseekv3_grpo_tutorial/grpo_config.yaml \
--sft_path_infer ./model_configs/deepseek_v3_config/predict_deepseek3_671b.yaml \
--sft_path_ref ./model_configs/deepseek_v3_config/ref_deepseek3_671b.yaml \
--sft_path_train ./model_configs/deepseek_v3_config/finetune_deepseek3_671b.yaml \
--tokenizer_path ./tokenizer.json \
--mind_dataset_dir ./output.mindrecord \
--use_parallel True \
--enable_compile_cache False" \
$worker_num $local_worker_num $master_ip 8118 $node_rank output/msrun_log False 7200

# 参数说明
# msrun 参数
worker_num:                   总卡数
local_worker_num:             单机的卡数
master_ip:                    主节点端口
node_rank:                    多机节点序号
# grpo_one_stage.py 参数
sft_path_infer:               推理用的模型配置文件
sft_path_ref:                 ref用的模型配置文件
sft_path_train:               训练用的模型配置文件
tokenizer_path:               模型对应的tokenizer文件tokenizer.json的路径
mind_dataset_dir:             训练数据集mindrecord文件的路径
save_data_file:               中间推理结果的保存路径(可选)
save_ckpt_dir:                训练ckpt的保存路径
use_parallel:                 是否并行
load_sft_checkpoint_infer:    推理模型(分布式)ckpt文件路径
load_sft_checkpoint_train:    训练模型(分布式)ckpt文件路径
load_ref_checkpoint:          参考模型(分布式)ckpt文件路径
load_ckpt_format:             加载权重格式，可选'ckpt'或'safetensors'
enable_compile_cache:         是否使用编译缓存
```

拉起任务后，通过以下命令查看运行日志

```shell
tail -f output/msrun/worker_0.log
```
