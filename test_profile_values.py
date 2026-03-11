#!/usr/bin/env python3
"""Test if profile tensor values can be retrieved from the encoder engine."""
import struct
import json
import numpy as np

pte_path = "/home/gasoonjia/trt/executorch/parakeet_tdt_exports/model.pte"

with open(pte_path, "rb") as f:
    data = f.read()

MAGIC = b"TR01"
HEADER_FORMAT = "<4sIIIQ8s"
HEADER_SIZE = 32

positions = []
start = 0
while True:
    pos = data.find(MAGIC, start)
    if pos == -1:
        break
    positions.append(pos)
    start = pos + 1

# Encoder is blob 1
pos = positions[1]
header_data = data[pos:pos + HEADER_SIZE]
magic, meta_offset, meta_size, engine_offset, engine_size, _ = \
    struct.unpack(HEADER_FORMAT, header_data)

eng_start = pos + engine_offset
eng_bytes = data[eng_start:eng_start + engine_size]

import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(eng_bytes)

print(f"Engine has {engine.num_io_tensors} IO tensors")
print(f"Engine has {engine.num_optimization_profiles} profiles")

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    location = engine.get_tensor_location(name)
    is_shape = (location == trt.TensorLocation.HOST)

    if mode != trt.TensorIOMode.INPUT:
        continue

    print(f"\n  Input '{name}' (is_shape={is_shape}):")

    if is_shape:
        # Try get_shape_input from profile
        try:
            profile = engine.get_optimization_profile_shape(name, 0)
            print(f"    get_optimization_profile_shape: {profile}")
        except Exception as e:
            print(f"    get_optimization_profile_shape: {e}")

        # Try get_tensor_profile_shape
        try:
            result = engine.get_tensor_profile_shape(name, 0)
            print(f"    get_tensor_profile_shape: {result}")
        except Exception as e:
            print(f"    get_tensor_profile_shape: {e}")
    else:
        try:
            result = engine.get_tensor_profile_shape(name, 0)
            print(f"    get_tensor_profile_shape: min={list(result[0])} opt={list(result[1])} max={list(result[2])}")
        except Exception as e:
            print(f"    get_tensor_profile_shape: {e}")

# Now check: can we build a context and infer output shapes?
print("\n=== Context shape inference test ===")
context = engine.create_execution_context()

# Compute shape tensor values at max (s18=5000)
shape_vals = {
    "sym_size": 5000, "add_1": 2500, "add_2": 1250,
    "add_3": 625, "sub_1": 4375, "add_5": 1249,
}

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode != trt.TensorIOMode.INPUT:
        continue

    location = engine.get_tensor_location(name)
    is_shape = (location == trt.TensorLocation.HOST)

    if name == "audio_signal":
        context.set_input_shape(name, (1, 128, 5000))
        print(f"  Set '{name}' shape (1,128,5000)")
    elif name in shape_vals and is_shape:
        # Set shape tensor value
        host_buf = np.array([shape_vals[name]], dtype=np.int32)
        context.set_input_shape(name, (1,))
        context.set_tensor_address(name, host_buf.ctypes.data)
        print(f"  Set shape tensor '{name}' = {shape_vals[name]}")
    elif name == "sub":
        # DEVICE input, not a shape tensor
        context.set_input_shape(name, (1,))
        print(f"  Set '{name}' shape (1,)")
    else:
        context.set_input_shape(name, (1,))
        print(f"  Set '{name}' shape (1,)")

# Query output shapes
print("\nOutput shapes:")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.OUTPUT:
        shape = context.get_tensor_shape(name)
        print(f"  '{name}' = {list(shape)}")
