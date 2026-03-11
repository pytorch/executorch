#!/usr/bin/env python3
"""Extract TRT engines from PTE and inspect them using TRT Python API."""
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

print(f"Found {len(positions)} TRT blob(s)")

# Focus on the encoder (blob 1 - the largest one with dynamic shapes)
encoder_idx = 1
pos = positions[encoder_idx]
header_data = data[pos:pos + HEADER_SIZE]
magic, meta_offset, meta_size, engine_offset, engine_size, _ = \
    struct.unpack(HEADER_FORMAT, header_data)

meta_start = pos + meta_offset
meta_json = data[meta_start:meta_start + meta_size].decode("utf-8")
meta = json.loads(meta_json)

print(f"\n=== Encoder io_bindings ===")
for b in meta["io_bindings"]:
    kind = "INPUT" if b["is_input"] else "OUTPUT"
    st = " SHAPE_TENSOR" if b.get("is_shape_tensor") else ""
    print(f"  {kind:6s} {b['name']:40s} dtype={b['dtype']:10s} shape={str(b['shape']):20s}{st}")

# Extract and load engine
eng_start = pos + engine_offset
eng_bytes = data[eng_start:eng_start + engine_size]

import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(eng_bytes)

print(f"\n=== Encoder Engine inspection ===")
num_io = engine.num_io_tensors
print(f"num_io_tensors = {num_io}")
print(f"num_optimization_profiles = {engine.num_optimization_profiles}")

for i in range(num_io):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    location = engine.get_tensor_location(name)
    is_shape = (location == trt.TensorLocation.HOST)

    mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
    loc_str = "HOST" if location == trt.TensorLocation.HOST else "DEVICE"
    dims = list(shape)

    print(f"\n  [{i}] {mode_str:6s} '{name}'")
    print(f"       dtype={dtype} shape={dims} loc={loc_str} is_shape={is_shape}")

    if mode == trt.TensorIOMode.INPUT:
        try:
            if is_shape:
                # Shape tensor: use profile values API
                # TRT 10.x API
                min_v = engine.get_tensor_profile_values(name, 0, trt.TensorProfileOperation.kMIN)
                opt_v = engine.get_tensor_profile_values(name, 0, trt.TensorProfileOperation.kOPT)
                max_v = engine.get_tensor_profile_values(name, 0, trt.TensorProfileOperation.kMAX)
                print(f"       profile values: min={list(min_v)} opt={list(opt_v)} max={list(max_v)}")
            else:
                # Regular tensor: use profile shape API
                min_s = engine.get_tensor_profile_shape(name, 0)
                print(f"       profile shapes: min={list(min_s[0])} opt={list(min_s[1])} max={list(min_s[2])}")
        except Exception as e:
            # Try alternate APIs
            try:
                ps = engine.get_profile_shape(0, name)
                print(f"       profile shapes (v2): min={list(ps[0])} opt={list(ps[1])} max={list(ps[2])}")
            except Exception as e2:
                print(f"       profile: error1={e}, error2={e2}")

# Try to create execution context and set shapes
print(f"\n=== Testing context shape inference ===")
context = engine.create_execution_context()

# Set shapes for trace-time (5000 frames)
# From the io_bindings, encoder inputs are:
# audio_signal [1,128,-1], length [1], sym_size [1], add_1 [1], add_2 [1],
# add_3 [1], sub [1], sub_1 [1], add_5 [1]
# For s18=5000: sym_size=5000, add_1=2500, add_2=1250, add_3=625,
# sub=4375 (=5000-625), sub_1=4375, add_5=1249
import torch
# Compute shape tensor values for s18=5000
s18 = 5000
sym_size = s18
add_1 = ((s18 - 1) // 2) + 1  # 2500
add_2 = ((add_1 - 1) // 2) + 1  # 1250
add_3 = ((add_2 - 1) // 2) + 1  # 625
sub = 4999 - (s18 - 1) // 8  # = 4999 - 624 = 4375
sub_1 = sub  # 4375
add_5 = add_3 + add_3 - 1  # 1249

print(f"Shape tensor values for s18={s18}:")
print(f"  sym_size={sym_size}, add_1={add_1}, add_2={add_2}, add_3={add_3}")
print(f"  sub={sub}, sub_1={sub_1}, add_5={add_5}")

# Set all inputs
shape_values = {
    "sym_size": sym_size,
    "add_1": add_1,
    "add_2": add_2,
    "add_3": add_3,
    "sub": sub,
    "sub_1": sub_1,
    "add_5": add_5,
}

for i in range(num_io):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    location = engine.get_tensor_location(name)
    is_shape = (location == trt.TensorLocation.HOST)

    if mode != trt.TensorIOMode.INPUT:
        continue

    if name == "audio_signal":
        context.set_input_shape(name, (1, 128, s18))
        print(f"  Set '{name}' input shape = (1, 128, {s18})")
    elif name == "length":
        context.set_input_shape(name, (1,))
        print(f"  Set '{name}' input shape = (1,)")
    elif name in shape_values:
        if is_shape:
            context.set_input_shape(name, (1,))
            print(f"  Set '{name}' shape tensor shape=(1,) value=[{shape_values[name]}]")
        else:
            # sub is NOT a shape tensor in engine but IS an input
            context.set_input_shape(name, (1,))
            print(f"  Set '{name}' regular input shape=(1,)")

# Check all shapes
print(f"\n  Output shapes after shape propagation:")
all_ok = True
for i in range(num_io):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.OUTPUT:
        shape = context.get_tensor_shape(name)
        dims = list(shape)
        print(f"    '{name}' = {dims}")
        if -1 in dims:
            print(f"    ERROR: unresolved!")
            all_ok = False

if all_ok:
    print("\n  All output shapes resolved successfully!")
else:
    print("\n  ERROR: Some output shapes not resolved!")
