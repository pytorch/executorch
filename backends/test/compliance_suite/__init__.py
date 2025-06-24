import os
import unittest

from enum import Enum
from typing import Any, Callable, Tuple

import logging
import torch
from executorch.backends.test.harness import Tester

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Read enabled backends from the environment variable. Enable all if
# not specified (signalled by None).
def get_enabled_backends():
    et_test_backends = os.environ.get("ET_TEST_BACKENDS")
    if et_test_backends is not None:
        return et_test_backends.split(",")
    else:
        return None

_ENABLED_BACKENDS = get_enabled_backends()

def is_backend_enabled(backend):
    if _ENABLED_BACKENDS is None:
        return True
    else:
        return backend in _ENABLED_BACKENDS

ALL_TEST_FLOWS = []

if is_backend_enabled("xnnpack"):
    from executorch.backends.xnnpack.test.tester import Tester as XnnpackTester

    XNNPACK_TEST_FLOW = ("xnnpack", XnnpackTester)
    ALL_TEST_FLOWS.append(XNNPACK_TEST_FLOW)

if is_backend_enabled("coreml"):
    from executorch.backends.apple.coreml.test.tester import CoreMLTester

    COREML_TEST_FLOW = ("coreml", CoreMLTester)
    ALL_TEST_FLOWS.append(COREML_TEST_FLOW)


DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
    torch.int32,
    torch.uint32,
    torch.int64,
    torch.uint64,
    torch.float16,
    torch.float32,
    torch.float64,
]

FLOAT_DTYPES =[
    torch.float16,
    torch.float32,
    torch.float64,
]

class TestType(Enum):
    STANDARD = 1
    DTYPE = 2

def dtype_test(func):
    setattr(func, "test_type", TestType.DTYPE)
    return func

def operator_test(cls):
    _create_tests(cls)
    return cls

def _create_tests(cls):
    for key in dir(cls):
        if key.startswith("test_"):
            _expand_test(cls, key)
        
def _expand_test(cls, test_name: str):
    test_func = getattr(cls, test_name)
    for (flow_name, tester_factory) in ALL_TEST_FLOWS:
        _create_test_for_backend(cls, test_func, flow_name, tester_factory)                        
    delattr(cls, test_name)

def _create_test_for_backend(
    cls,
    test_func: Callable,
    flow_name: str,
    tester_factory: Callable[[torch.nn.Module, Tuple[Any]], Tester]
):
    test_type = getattr(test_func, "test_type", TestType.STANDARD)

    if test_type == TestType.STANDARD:
        def wrapped_test(self):
            test_func(self, tester_factory)
            
        test_name = f"{test_func.__name__}_{flow_name}"
        setattr(cls, test_name, wrapped_test)
    elif test_type == TestType.DTYPE:
        for dtype in DTYPES:
            def wrapped_test(self):
                test_func(self, dtype, tester_factory)
            
            dtype_name = str(dtype)[6:] # strip "torch."
            test_name = f"{test_func.__name__}_{dtype_name}_{flow_name}"
            setattr(cls, test_name, wrapped_test)
    else:
        raise NotImplementedError(f"Unknown test type {test_type}.")


class OperatorTest(unittest.TestCase):
    def _test_op(self, model, inputs, tester_factory, use_random_test_inputs=True):
        tester = (
            tester_factory(
                model,
                inputs,
            )
            .export()
            .to_edge_transform_and_lower()
        )

        is_delegated = any(
            n.target == torch._higher_order_ops.executorch_call_delegate
            for n in tester.stages[tester.cur].graph_module.graph.nodes
            if n.op == "call_function"
        )

        # Only run the runtime test if the op was delegated.
        if is_delegated:
            (
                tester
                .to_executorch()
                .serialize()
                # If use_random_test_inputs is False, explicitly pass the export inputs for the test.
                # This is useful for ops like embedding where the random input generation isn't aware
                # of the constraints on the input data values (e.g. the indices must be within bounds).
                # If use_random_test_inputs is True, a value of None is passed to cause test inputs to
                # be randomly generated.
                .run_method_and_compare_outputs(inputs = inputs if not use_random_test_inputs else None)
            )
    