# W4A16 v8 role-aware integration package

Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent).

V8 carries five v7 schedules measured in the independently validated 72-pair
RTX5090 report. It adds a small pre-trace policy, not new arithmetic or a new
kernel timing claim. The packed `CudaCoalescedInt4Tensor` fields, FP32 affine
reconstruction, final BF16 weight operand, BF16 activation/output, FP32 MMA and
deterministic split reduction are unchanged.

Accepted exact-SM120 actual-GGUF routes are Gemma attention M8/M16, down M8 and
fused gate/up M16. Vocabulary, fused M8, down M16, all SM80 calls, unsupported
shapes and buckets M1/M2/M4 use the supplied unchanged native callable. Synthetic
MG M8 suggestions require an explicit opt-in and are not accepted deployment
routes.

`bind_launcher` derives the real CUDA capability and resolves shape/config before
tracing. Candidate execution is restricted to M5-8 or M9-16; M1/M2/M4 and any
contract mismatch forward the original tensors unchanged to native. Accepted
routes require a measured ratio below 0.98.

The traceable `triton_op` boundary passed fullgraph dynamic `torch.compile` and
`torch.export` for every accepted route with a literal config index. Local AOTI
packaging remains unvalidated because the environment lacks the CUDA driver
header `cuda.h`. Parent AOTI package/load, server/F1/performance and memory
validation remain mandatory.
