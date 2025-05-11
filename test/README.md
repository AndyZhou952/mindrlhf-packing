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

```
-    real_sample_num = ops.sum(sample_valid_len != 1, dtype=mstype.int32)
+    # when there's only one sample per batch, average over all bs entries
+    if pack_sample_num == 1:
+        real_sample_num = Tensor(batch_size, mstype.int32)
+    else:
+        # for true packing you may want to count only segments with >0 length
+        real_sample_num = ops.sum(sample_valid_len > 0, dtype=mstype.int32)
```

Dummy mask fix (more robust)
```
# in pack_grouped_data_new:
dummy_mask = np.zeros(total_samples, dtype=int)
# for real i in [0..real_sample_num-1]: dummy_mask[i] = 1
# for dummies i in [real..]:          dummy_mask[i] = 0

return {
  ...,
  "dummy_mask": dummy_mask,      # 1 for real slots, 0 for dummy
}

def construct(
    ...,
    sample_valid_len,    # [bs, packed_sample_num]
    dummy_mask,          # [bs, packed_sample_num], 1=real,0=dummy
):
    ...
    deno = ...
    nume = sample_valid_len

    dm = self.cast(dummy_mask, deno.dtype)        # [bs,packed]
    per_slot = (deno / nume) * dm                 # zero out dummy slots
    real_sample_num = dm.sum(dtype=mstype.int32)  # count only the real ones
    loss = per_slot.sum() / real_sample_num
    return loss
```
