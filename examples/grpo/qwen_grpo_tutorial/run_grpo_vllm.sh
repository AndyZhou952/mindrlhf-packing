export PYTHONPATH=/path/to/mindrlhf:/path/to/mindformers/:$PYTHONPATH
export GLOG_v=2

export vLLM_MODEL_BACKEND=MindFormers
export vLLM_MODEL_MEMORY_USE_GB=40
export MINDFORMERS_MODEL_CONFIG=/path/to/mindrlhf/model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research


msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
./main.py \
--config ./grpo_config.yaml \
--sft_path_infer /path/to/mindrlhf/model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml \
--sft_path_train /path/to/mindrlhf/model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml \
--vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--mind_dataset_dir /path/to/limr_template_qwenr1.mindrecord \
--save_data_file /path/to/grpo_1024.mindrecord \
--save_ckpt_dir /path/to/train \
--use_parallel True \
--enable_compile_cache False \
--load_sft_checkpoint_infer "/path/to/qwen25_7b" \
--load_sft_checkpoint_train "/path/to/qwen25_7b" \
--load_ref_checkpoint "/path/to/qwen25_7b" > vllm.log 2>&1 &
