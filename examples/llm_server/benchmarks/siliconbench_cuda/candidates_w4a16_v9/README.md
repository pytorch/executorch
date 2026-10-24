# W4A16 v9 native-AOTI-matched portfolio

Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent).

V9 contains ten pending RTX5090 schedules focused on Gemma down/fused and MG
M8/M16 shapes. It matches measured native outer tiles and tests only bounded
split-count variants. The accepted v8 attention-M8 schedule is retained as a
control. No v9 route is automatically enabled before fresh actual-AOTI SM120
validation.

The six-tensor packed ABI, K32 scale/zero metadata, FP16 K256 steps, FP32 affine
reconstruction, final BF16 weight operand, BF16 activation/output and FP32 MMA
are unchanged. No repack, persistent dense cache, A8/FP8 path, new bucket, bias
fusion, or SM100-only instruction is used. Bias remains an external parent
epilogue and is validated separately.

`candidate_op(x,qdata,scale,scale_step,zero,zero_step,config_index)` keeps the v8
trace/export ABI with M5-8 and M9-16 bounds. Production policy remains native on
all devices until the parent accepts a v9 route. M1/M2/M4 remain W4A8.
