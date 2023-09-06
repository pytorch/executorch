from functools import partial
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.optim as optim
from typing import List, Tuple, Union, Optional
import executorch.exir as exir

# Mutate on any input does not trace
def func(k):
    k[1] = torch.ones(1,4)
    return k
k = torch.rand(2,4)
print(func(k))
input_tuple = (k, )

constraints = []
captured = exir.capture(
    func, input_tuple, exir.CaptureConfig(pt2_mode=True, enable_aot=True), constraints=constraints # enable_aot=False works
    )
print(captured.exported_program.graph)

# Below code does not work when enable_aot=True, with error,
# File
# "/Users/myuan/miniconda3/envs/executorch/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line
# 3342, in create_aot_dispatcher_function
# raise RuntimeError(f"""
#     RuntimeError:
#     Found following user inputs located at [0, 1] are mutated. This is currently banned in the aot_export workflow.
#     If you need this functionality, please file a github issue.
# def func(k, v):
#     # k is a list. i'th item is a tensor in ith layer.
#     # for tensor k[i], it's leading dim is token index.
#     n_layer = len(k)
#     n_tok = k[0].size(0)
#     for j in range(n_tok):
#         for i in range(n_layer):
#             k[i][j] = torch.ones(1, 2)
#     return k, v
#
# k = [torch.rand(3,2), torch.rand(3,2)]
# v = [torch.rand(3,2), torch.rand(3,2)]
# print(func(k, v))
#
# import executorch.exir as exir
# input_tuple = (k, v)
# # constraints = [torch._export.dynamic_dim(x, 0) >= 1, torch._export.dynamic_dim(x, 0) <= 100]
# constraints = []
# captured = exir.capture(
#     func, input_tuple, exir.CaptureConfig(pt2_mode=True, enable_aot=True), constraints=constraints
#     )
# print(captured.exported_program.graph)
