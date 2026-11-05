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
