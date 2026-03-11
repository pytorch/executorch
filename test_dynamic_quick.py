#!/usr/bin/env python3
"""Quick test of dynamic TRT encoder with various input shapes."""
import torch
import numpy as np
from executorch.runtime import Runtime

pte_path = "/home/gasoonjia/trt/executorch/parakeet_tdt_exports/model.pte"
print(f"Loading: {pte_path}")
runtime = Runtime.get()
program = runtime.load_program(pte_path)

# Test 1: trace-time shape (5000 frames)
print("\n=== Test 1: trace-time shape (5000 frames) ===")
mel = torch.randn(1, 128, 5000, dtype=torch.float32)
mel_len = torch.tensor([5000], dtype=torch.int64)
encoder = program.load_method("encoder")
try:
    result = encoder.execute([mel, mel_len])
    print(f"  Output shape: {result[0].shape}, enc_len={result[1].item()}")
    print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: smaller shape (744 frames, like real_speech.wav)
print("\n=== Test 2: smaller shape (744 frames) ===")
mel2 = torch.randn(1, 128, 744, dtype=torch.float32)
mel_len2 = torch.tensor([744], dtype=torch.int64)
encoder2 = program.load_method("encoder")
try:
    result2 = encoder2.execute([mel2, mel_len2])
    print(f"  Output shape: {result2[0].shape}, enc_len={result2[1].item()}")
    print(f"  Output range: [{result2[0].min():.4f}, {result2[0].max():.4f}]")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: minimum shape (161 frames)
print("\n=== Test 3: minimum shape (161 frames) ===")
mel3 = torch.randn(1, 128, 161, dtype=torch.float32)
mel_len3 = torch.tensor([161], dtype=torch.int64)
encoder3 = program.load_method("encoder")
try:
    result3 = encoder3.execute([mel3, mel_len3])
    print(f"  Output shape: {result3[0].shape}, enc_len={result3[1].item()}")
    print(f"  Output range: [{result3[0].min():.4f}, {result3[0].max():.4f}]")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
