# Debugging Notes

### GRPOModel Construct
```python
    pack_sample_num = sample_valid_len.shape[1]
    real_sample_num = ops.sum(sample_valid_len != 1, dtype=mstype.int32)

    # ── DEBUG LOGGING ─────────────────────────────────────────────────────────────
    # 1) sample_index should all be zero for pack_sample_num=1
    print("sample_index unique:", np.unique(sample_index.asnumpy()))
    # 2) sample_valid_len vs responses_mask.sum
    #    note: in mindspore you may need to cast to numpy to sum on CPU:
    mask_counts = responses_mask.sum(axis=1).asnumpy()  # [bs]
    valid_counts = sample_valid_len[:, 0].asnumpy()    # [bs]
    print("responses_mask.sum:", mask_counts)
    print("sample_valid_len[:,0]:", valid_counts)
    # 3) real_sample_num vs batch_size
    bs = prompt_completion_ids.shape[0]
    print(f"real_sample_num={real_sample_num.asnumpy()}  batch_size={bs}")
    # ─────────────────────────────────────────────────────────────────────────

```
