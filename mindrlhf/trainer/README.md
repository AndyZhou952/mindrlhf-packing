### Key changes in decoupling:
- (1) in `grpo_config.py`, packing_sample_length is never used. Also removed the initialization part in `grpo_trainer.py`
- (2) in `_make_experience`, set `pack_num = 1` if `packing = False` (doing it this way is easier, as we can keep the loss definition in `grpo_models.py` unchanged).
- (3) moved the packing-related algorithm in a separate util file, `packing_utils.py` (currently under mindrlhf/trainer).
- (4) added `seq_length` and `pad_token_id` in the function header of pack_grouped_data (substituting `self.grpo_config.seq_length` and `self.tokenizer.eos_token_id`), similar for the functions `pack_grpo_data` and `create_pack_group`.
- (5) rewrote `construct_inputs_packing`, `pack_grpo_data`, `pack_grouped_data` for readability & performance optimization. They are equivalent to the original implementation and faster. One may find the test cases and optimization results [here](https://github.com/AndyZhou952/grpo-packing).
