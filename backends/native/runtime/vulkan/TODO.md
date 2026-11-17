# Vulkan engine TODOs

## Plan conversion temporaries in the shared memory plan

Tensors that `VulkanEngineExecutable.cpp` creates for storage or layout
conversions call `graph_->add_tensor` without a `memory_plan_` allocation id,
so each gets dedicated device memory instead of sharing a planned slot:

- `repack_to_width`, including the SDPA query conversion to buffer storage
- the `aten.view_copy.default` storage clone
- the channels-packed `aten.embedding.default` temporary

HuggingFace Llama 3.2 1B 8da4w, 2K export: peak GPU memory is 2794 MiB, against
2265 MiB with buffer SDPA disabled and 2263 MiB for the ET-VK `.pte`. The
+528 MiB comes from enabling buffer SDPA. Give these temporaries lifetimes in
the memory plan, measure again, and attribute whatever remains.

## Lower int4 weight-only linear for any N

`FuseQuantizedLinear` leaves an int4 weight-only linear unlowered when N is not
a multiple of 4, to match `VK_CHECK_COND(N % 4 == 0)` in ET-VK's
`prepack_q4_w_4x8_nc_buffer`, and the engine then rejects the quantized weight.
Round N up to a multiple of 4 there (the pack shader already fills n >= N with
zeros) and mask the partial last tile on store. Until then Vulkan fails the op
sample tests `linear_int4wo_*.linear_feature_sizes_2` and
`linear_int4wo_*.linear_unbatched_0`, where N is 1 and 43.

## Rank-4 linear input

Vulkan returns wrong outputs for the op sample test
`linear_float.linear_leading_batch_1`, an `nn.Linear(67, 43)` on a
(2, 4, 8, 67) input: in f32 the fitted rel_bias is -0.48, and f16 returns
values up to 6.1e4. Both are consistent with only one of the two outer batches
being written. The rank-3 inputs (4, 8, 67) and (2, 7, 256) pass. Find where
the outer dimension is lost: the engine's linear lowering, or ET-VK's linear on
a rank-4 tensor.
