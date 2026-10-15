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

## Whisper: zero-copy tensor sharing and fused-SDPA weight storage

`export_hf_whisper` (backends/native/examples/hf) exports encoder, cross_kv
and static-cache decoder methods; `LowerHFAttention` handles the decoder's
learned-position (no-RoPE) blocks. Cross K/V are projected once by cross_kv
and staged as decoder inputs, but they still round-trip through the host
between executables (as does the encoder output before them). Zero-copy
chaining needs the EngineContext contract implemented here: matching
DataBinding keys share one context-owned buffer across executables (see the
MutableStateClaim TODO in VulkanEngineExecutable.cpp), or an API to bind one
executable's output tensor as another's input.

Whisper's encoder makes the fused-SDPA attention weights [1, H, 1500, 1500]
fp32 (~54 MiB per layer at H=6). They follow the output's storage, so they
land in a texture unless the output exceeds texture limits; measure on a
device and consider forcing buffer storage for large-S fused SDPA, as
`buffer_attention_fits` does for `llama.custom_sdpa`.
