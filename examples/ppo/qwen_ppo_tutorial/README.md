# Qwen-PPO教程

本教程基于Qwen2引导读者跑通PPO训推一体的基本流程。

## 环境准备

| 依赖        | 推荐版本                 |
| ----------- | ------------------------ |
| CANN        | 7.5 (C20_20241024)       |
| MindSpore   | 2.5.0 (12月24日)         |
| MindFormers | 1.3.0 (dev分支 12月26日) |

## 推理模型权重转换

用户可以从[HuggingFace官方](https://huggingface.co/Qwen/Qwen2-7B-Instruct)或[魔搭社区](https://modelscope.cn/models/Qwen/Qwen2-7B)下载完整预训练权重，`vocab.json`和`merges.txt`文件也可在上述链接中下载。
下载完成后，运行如下[转换脚本](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2/convert_weight.py)，将权重转换为完整的ckpt权重。

```shell
python convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path {path}/MS_CKPT_NAME --dtype bf16

model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

由于下载的通常为单卡完整权重，若推理采用分布式策略，则需要进行[权重转换](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)，将单卡完整权重转化为分布式权重。由于当前推理采用前端推理策略，因此在转换权重时，只需要将设置模型切分策略，数据切分策略设置为1，如当model_parallel(mp): 4, data_parallel(dp): 2 时，在转换权重时分布式策略为mp4dp1，随后会得到rank_0至rank_3 4份权重，随后将rank_0至rank_3 4份权重复制并重新命名为rank_4至rank_7。权重对应关系为：rank_0 &rarr; rank_4、rank_1 &rarr; rank_5、rank_2 &rarr; rank_6、rank_3 &rarr; rank_7。

## 数据预处理

采用cvalues数据集进行预处理，命令如下：

```sh
cd examples/qwen_ppo_tutorial
python rlhf_data.py \
 --vocab_path path/to/vocab.json \
 --merges_file_path path/to/merges.txt \
 --file_path path/to/input.jsonl \
 --output_path path/to/output.mindrecord \
 --max_prompt_length 4096 \
 --seq_length 8193
```

## 运行命令

```sh
msrun --worker_num=4 --local_worker_num=4 \
 --master_addr=127.0.0.1 --master_port=9123 \
 --join=True --log_dir=./qwen2_one_log \
 examples/qwen_ppo_tutorial/ppo_one_stage.py \
 --sft_path_infer ./model_configs/qwen_ppo/generate_qwen2_7b.yaml \
 --sft_path_train ./model_configs/qwen_ppo/finetune_qwen2_7b.yaml \
 --reward_path ./model_configs/qwen_ppo/generate_qwen2_7b.yaml \
 --critic_path ./model_configs/qwen_ppo/generate_qwen2_7b.yaml \
 --mind_dataset_dir path/to/output.mindrecord \
 --save_ckpt_dir path/to/save_ckpt \
 --use_parallel True \
 --load_sft_checkpoint_infer path/to/infer_ckpt \
 --load_sft_checkpoint_train path/to/train_ckpt \
 --enable_compile_cache False \
 --only_save_strategy False

# 参数说明
# only_save_strategy: 若要获取各模型的分布式策略文件，设置为True。程序将编译模型，得到策略文件并保存到 ./strategy 目录，然后直接退出。
```

​    PPOTrainer中有ppo_model_infer和ppo_model_train两个成员变量，两者分布式策略不同，加载不同的yaml进行初始化。

