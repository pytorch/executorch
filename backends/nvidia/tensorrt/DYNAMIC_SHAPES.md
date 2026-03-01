# Fix expand_copy for TRT dynamic shapes via shape tensor API

## Context

The `expand_copy` converter passes `-1` to ISliceLayer's output shape for
dynamic dims.  TRT rejects this: "inherently negative length — proven upper
bound is -1."  This blocks 161 expand ops (attention mask broadcasting) from
TRT delegation, fragmenting the encoder into many small partitions.

**Fix:** compute the output shape dynamically at runtime using TRT's shape
tensor API (`add_shape` → `add_gather` → `add_concatenation` → `set_input`).

Steps 1–3 from the previous plan (C++ runtime, `setInputShape`, buffer
allocation) are already committed.  The remaining blocker is this converter.

## File to modify

[expand.py](backends/nvidia/tensorrt/converters/expand.py) — `convert_expand`
(lines 97–191)

## Implementation

Replace the slice-layer block (lines 164–191) with a two-path branch:

### Static path (no `-1` in output_shape) — keep as-is

```python
if all(d >= 0 for d in output_shape):
    # ... existing add_slice with trt.Dims(shape) ...
```

### Dynamic path (any `-1` in output_shape) — new

```python
import tensorrt as trt
import numpy as np

# 1. Get input runtime shape as int32 tensor
shape_layer = network.add_shape(current_tensor)
shape_layer.name = f"expand_shape_{node.name}"
shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
shape_i32.name = f"expand_shape_i32_{node.name}"

# 2. Build output shape tensor, one component per dim
components = []
for i, (inp_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
    if out_dim >= 0:
        # Concrete dim → constant [out_dim]
        c = network.add_constant([1], trt.Weights(np.array([out_dim], np.int32)))
        c.name = f"expand_dim{i}_{node.name}"
        components.append(c.get_output(0))
    elif inp_dim == 1:
        # Broadcasting from 1 → dynamic target.
        # The target came from expand_size[i] which is an FX Node.
        # Get the corresponding TRT tensor from input_map.
        raw_target = expand_size[i]
        if isinstance(raw_target, torch.fx.Node) and raw_target in input_map:
            t = input_map[raw_target]
            shuf = network.add_shuffle(t)
            shuf.reshape_dims = trt.Dims([1])
            shuf.name = f"expand_target{i}_{node.name}"
            cast = network.add_cast(shuf.get_output(0), trt.int32)
            cast.name = f"expand_target_i32_{i}_{node.name}"
            components.append(cast.get_output(0))
        else:
            # Fallback: extract from input shape (same dim, no broadcast)
            idx = network.add_constant([1], trt.Weights(np.array([i], np.int32)))
            g = network.add_gather(shape_i32.get_output(0), idx.get_output(0), axis=0)
            g.name = f"expand_gather{i}_{node.name}"
            components.append(g.get_output(0))
    else:
        # Dynamic dim, no broadcast → extract from input shape
        idx = network.add_constant([1], trt.Weights(np.array([i], np.int32)))
        g = network.add_gather(shape_i32.get_output(0), idx.get_output(0), axis=0)
        g.name = f"expand_gather{i}_{node.name}"
        components.append(g.get_output(0))

# 3. Concatenate into 1-D shape tensor
shape_cat = network.add_concatenation(components)
shape_cat.axis = 0
shape_cat.name = f"expand_outshape_{node.name}"
shape_tensor = shape_cat.get_output(0)

# 4. Build stride tensor (0 = broadcast, 1 = keep)
stride_np = np.array(stride, dtype=np.int32)
stride_const = network.add_constant([len(stride)], trt.Weights(stride_np))
stride_const.name = f"expand_stride_{node.name}"

# 5. Build start tensor (all zeros)
start_np = np.zeros(output_dims, dtype=np.int32)
start_const = network.add_constant([output_dims], trt.Weights(start_np))
start_const.name = f"expand_start_{node.name}"

# 6. Create slice layer with set_input overrides
slice_layer = network.add_slice(
    current_tensor, start=[0]*output_dims, shape=[1]*output_dims, stride=[1]*output_dims
)
slice_layer.set_input(1, start_const.get_output(0))
slice_layer.set_input(2, shape_tensor)
slice_layer.set_input(3, stride_const.get_output(0))
slice_layer.name = f"expand_slice_{node.name}"
```

Key details:
- `add_shape` returns int64; must cast to int32 for `add_concatenation`
- `add_gather` with axis=0 extracts a single dim from the shape vector
- `set_input(2, ...)` overrides the static shape placeholder — TRT can now
  prove the dims are positive via the optimization profile

## Verification

```bash
python examples/models/parakeet/export_parakeet_tdt.py \
    --backend tensorrt --output-dir ./parakeet_tensorrt
```

Expected: export succeeds with dynamic shapes; encoder is a single TRT
delegate (no expand_copy fallback).  Then test transcription with 7.4s
audio to verify it produces the same text as eager/CUDA.
