import numpy as np
from mindformers import logger

def pad_sequence_to_length(sequence: np.ndarray, target_length: int, pad_value: int) -> np.ndarray:
    """Pad a 1D sequence to the target length with a constant pad value."""
    current_length = len(sequence)
    if current_length < target_length:
        return np.pad(sequence, (0, target_length - current_length),
                      mode="constant", constant_values=pad_value)
    return sequence[:target_length]

def create_pack_group(data_dict_list: list, pack_num: int, seq_length: int) -> list:
    """Group samples into packs of up to `pack_num` sequences, without exceeding max length (seq_length)."""
    pack_group = []
    each_group = []
    current_group_length = 0

    for data in data_dict_list:
        sample_length = data["response_end_index"] - data["prompt_start_idx"] + 2
        needed_length = current_group_length + sample_length + (pack_num - len(each_group) - 1)
        if len(each_group) >= pack_num or needed_length > seq_length:
            pack_group.append(each_group)
            each_group = []
            current_group_length = 0
        each_group.append(data)
        current_group_length += sample_length

    if each_group:
        pack_group.append(each_group)
    return pack_group

def pack_grouped_data(pack_list: list, pack_num: int, seq_length: int, pad_token_id: int) -> dict:
    """Pack a single group of samples into one concatenated sequence with padding."""
    real_sample_num = len(pack_list)
    dummy_sample_num = pack_num - real_sample_num
    pad_to_length = seq_length - dummy_sample_num

    total_sequence_slots = seq_length
    total_samples = real_sample_num + dummy_sample_num

    # pre-allocate arrays
    prompt_completion_ids = np.full(total_sequence_slots, pad_token_id, dtype=int)
    responses_mask = np.zeros(total_sequence_slots, dtype=int)
    advantages = np.zeros(total_sequence_slots, dtype=float)
    sample_index = np.zeros(total_sequence_slots, dtype=int)
    actual_sequence_length = np.zeros(total_samples, dtype=int)
    sample_valid_length = np.zeros(total_samples, dtype=int)

    occupied_length = 0

    for i, data_dict in enumerate(pack_list):
        sample_prompt_completion_ids = data_dict["prompt_completion_ids"]
        sample_response_mask = data_dict["response_mask"]
        sample_advantage_value = data_dict["advantage"]
        prompt_start_idx = data_dict["prompt_start_idx"]
        response_end_index = data_dict["response_end_index"]

        original_length = response_end_index - prompt_start_idx + 2

        segment = sample_prompt_completion_ids[prompt_start_idx : response_end_index + 1]
        tmp_prompt_ids = pad_sequence_to_length(segment, original_length, pad_token_id)
        mask_segment = sample_response_mask[prompt_start_idx : response_end_index + 1]
        tmp_responses_mask = pad_sequence_to_length(mask_segment, original_length, 0)

        tmp_sample_index = np.full(original_length, i, dtype=int)
        tmp_advantages = np.full(original_length, sample_advantage_value, dtype=float)


        if i == real_sample_num - 1:
            tail_length = pad_to_length - occupied_length
            tmp_prompt_ids = pad_sequence_to_length(tmp_prompt_ids, tail_length, pad_token_id)
            tmp_responses_mask = pad_sequence_to_length(tmp_responses_mask, tail_length, 0)
            tmp_advantages = pad_sequence_to_length(tmp_advantages, tail_length, 0)
            tmp_sample_index = pad_sequence_to_length(tmp_sample_index, tail_length, i)

            write_length = tail_length
            actual_sequence_length[i] = pad_to_length
        else:
            write_length = original_length
            actual_sequence_length[i] = occupied_length + original_length

        prompt_completion_ids[occupied_length : occupied_length + write_length] = tmp_prompt_ids
        responses_mask      [occupied_length : occupied_length + write_length] = tmp_responses_mask
        advantages          [occupied_length : occupied_length + write_length] = tmp_advantages
        sample_index        [occupied_length : occupied_length + write_length] = tmp_sample_index

        sample_valid_length[i] = int(tmp_responses_mask.sum())
        occupied_length += write_length

    # fill dummy, prompt completion ids already pad_token_id, responses_mask and advantages already zero
    start = occupied_length
    end   = start + dummy_sample_num
    sample_index[start:end] = np.arange(real_sample_num, real_sample_num + dummy_sample_num, dtype=int)
    base_length = actual_sequence_length[real_sample_num - 1]
    actual_sequence_length[real_sample_num : real_sample_num + dummy_sample_num] = \
        base_length + np.arange(1, dummy_sample_num + 1, dtype=int)
    sample_valid_length[real_sample_num : real_sample_num + dummy_sample_num] = 1

    result = {
        "prompt_completion_ids": prompt_completion_ids,
        "responses_mask":        responses_mask,
        "advantages":            advantages,
        "actual_sequence_length": actual_sequence_length,
        "sample_index":          sample_index,
        "sample_valid_length":   sample_valid_length,
    }

    return result

def pack_grpo_data(prompt_completion_ids: np.ndarray,
                   prompts_mask: np.ndarray,
                   responses_mask: np.ndarray,
                   advantages: np.ndarray,
                   pack_num: int,
                   seq_length: int,
                   pad_token_id: int) -> list:
    """Entry point to pack RLHF data. raw samples --> packed sequences."""
    bs, seq_len = prompts_mask.shape
    advantages = advantages.reshape(-1)
    logger.info(f"advantages shape in pack: {advantages.shape}")

    # determine if prompts and responses are non-empty
    has_prompt   = prompts_mask.any(axis=1)
    has_response = responses_mask.any(axis=1)

    # warnings
    zero_prompts   = np.where(~has_prompt)[0]
    zero_responses = np.where(has_prompt & ~has_response)[0]
    if zero_prompts.size > 0:
        logger.warning(
            "prompts_mask is all zero for indices [%s]!",
            ", ".join(map(str, zero_prompts.tolist()))
        )
    if zero_responses.size > 0:
        logger.warning(
            "responses_mask is all zero for indices [%s]!",
            ", ".join(map(str, zero_responses.tolist()))
        )

    # identify prompt_start_idx and response_end_index
    first_prompt   = prompts_mask.argmax(axis=1)
    last_from_end  = np.flip(responses_mask, axis=1).argmax(axis=1)
    last_response  = seq_len - 1 - last_from_end

    # keep only those with both prompt and response
    valid_idx = np.where(has_prompt & has_response)[0]

    data_dict_list = [
        {
            "prompt_completion_ids": prompt_completion_ids[i],
            "prompt_mask":           prompts_mask[i],
            "response_mask":         responses_mask[i],
            "advantage":             float(advantages[i]),
            "prompt_start_idx":      int(first_prompt[i]),
            "response_end_index":    int(last_response[i]),
        }
        for i in valid_idx
    ]

    pack_groups = create_pack_group(data_dict_list, pack_num, seq_length)
    result      = [
        pack_grouped_data(group, pack_num, seq_length, pad_token_id)
        for group in pack_groups
    ]
    return result

def construct_inputs_packing(all_packed: list, batch_size: int, idx: int):
    """Construct inputs for packing."""
    start = idx * batch_size
    end = start + batch_size
    batch_slice = all_packed[start:end]

    input_id_batch = np.array([pack["prompt_completion_ids"] for pack in batch_slice], dtype=np.int32)
    actual_seq_length_batch = np.array([pack["actual_sequence_length"] for pack in batch_slice], dtype=np.int32)
    return input_id_batch, actual_seq_length_batch

