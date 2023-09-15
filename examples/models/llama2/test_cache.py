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
# from executorch.extension.pybindings.aten_lib_plus_custom import (
#     _load_bundled_program_from_buffer,
#     _load_for_executorch_from_buffer,
#     _load_for_executorch_from_bundled_program,
# )
import executorch.exir as exir

# # Mutate on any input does not trace
# def func(k):
#     k [1] = torch.ones(1,4)
#     return k
# k = torch.rand(2,4)
# print(func(k))
# input_tuple = (k, )

# cat traces, since there's a copy
def func(x, cache):
    k = x + cache[0]
    cache = torch.cat((cache, k))
    return cache

x = torch.ones(1,4)
cache = torch.rand(2,4)
print(func(x, cache))
input_tuple = (x, cache)

# constraints = [dynamic_dim(t, 0) <= 256,]
constraints = []
captured = exir.capture(
    func, input_tuple, exir.CaptureConfig(pt2_mode=True, enable_aot=True), constraints=constraints # enable_aot=False works
    )
print(captured.exported_program.graph)

edge = captured.to_edge()
print(edge.exported_program)
from executorch.exir.passes import MemoryPlanningPass
config = exir.ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    memory_planning_algo="greedy",
                    # allow_lifetime_and_storage_overlap: bool = False,
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                )
            )

exe_prog = edge.to_executorch(config)
print(exe_prog.exported_program)
# exir.print_program.pretty_print(exe_prog.program.execution_plan)

