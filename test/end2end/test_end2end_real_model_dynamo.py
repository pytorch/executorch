# flake8: noqa: F401
import copy
import unittest
from typing import Dict, Tuple

import executorch.exir as exir

# @manual=//executorch/extension/pytree:pybindings
import executorch.extension.pytree as pytree
import torch
from executorch.exir import (
    CaptureConfig,
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    memory,
)
from executorch.exir.passes import (
    DebugPass,
    MemoryPlanningPass,
    to_scratch_op_pass,
    ToOutVarPass,
)
from executorch.exir.tensor import make_tensor_value, TensorSpec
from executorch.exir.tracer import using_dynamo

# pyre-fixme[21]: Could not find module `executorch.extension.pybindings.portable`.
from executorch.extension.pybindings.portable import _load_for_executorch_from_buffer
from torch import nn


def allclose(lhs, rhs, rtol=1e-5, atol=1e-8):
    r"""
    Unlike torch.allocse which only handles Tensor arguments, allclose handles
    list, tuple, dict and nesting of these as well.
    """
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return torch.allclose(lhs, rhs, rtol, atol)
    if isinstance(lhs, (tuple, list)) and isinstance(rhs, (tuple, list)):
        return len(lhs) == len(rhs) and all(
            allclose(a, b, rtol, atol) for a, b in zip(lhs, rhs)
        )
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        lhs_keys = set(lhs.keys())
        rhs_keys = set(rhs.keys())
        if lhs_keys != rhs_keys:
            return False
        return all(allclose(lhs[k], rhs[k], rtol, atol) for k in lhs)
    else:
        raise RuntimeError(
            f"Unexpected types: lhs type {type(lhs)}, rhs type {type(rhs)}"
        )


class ExecutorchDynamoTests(unittest.TestCase):
    @unittest.skip(
        "temporarily disable this since this test needs some revisit after dynamo change."
    )
    def test_end_to_end_executorch_dynamo(self):
        import contextlib

        import torch
        import torch.utils._pytree as pytree
        from pye.model_inventory.asr_models.milan_dictation.MilanDictationModel import (
            MilanDictationModelGen,
        )

        print("Imports finished")

        sm = MilanDictationModelGen()
        model = sm.get_eager_model()

        print("Model loaded")

        reps = list(sm.get_representative_inputs_for("encode", 1))
        inputs = copy.deepcopy(reps[0])

        # TODO (this doesn't work with pt2_mode=True) with following error:
        # It appears that you're trying to get value out of a tracing tensor - erroring out! It's likely that this is caused by data-dependent control flow or similar.
        # graph_module = exir.trace(model.encode, inputs, pt2_mode=True)
        with using_dynamo(True):
            edge_dialect_gm = exir.capture(
                model.encode, inputs, CaptureConfig(pt2_mode=False)
            ).to_edge()

        inputs = copy.deepcopy(reps[0])
        flat_args, in_spec = pytree.tree_flatten(inputs)

        final_graph_module = edge_dialect_gm.to_executorch(
            ExecutorchBackendConfig(
                passes=[
                    DebugPass("Before converting to out variant"),
                    ToOutVarPass(True),
                    DebugPass("After converting to out variant"),
                    to_scratch_op_pass,
                ],
                memory_planning_pass=MemoryPlanningPass("greedy"),
            )
        ).dump_graph_module()
        x = final_graph_module(*inputs)
