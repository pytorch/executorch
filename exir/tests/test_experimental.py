# pyre-strict

import copy
import unittest
from typing import Optional

import executorch.exir as exir
import executorch.exir.tests.models as models

import torch
from executorch.exir import CaptureConfig
from executorch.exir.error import ExportError
from executorch.exir.experimental import (
    add_assertions,
    convert_fake_tensor_to_tensor_meta,
)
from executorch.exir.experimental.export_pt2 import (
    ExportSession,
    Guard,
    GuardResolution,
    GuardType,
    Trace,
    trace,
)
from functorch.experimental import control_flow
from torch._subclasses.fake_tensor import FakeTensor


class TestExperimental(unittest.TestCase):
    def test_assertion_inserts(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.sin(x)
            return torch.add(x, y)

        x = (torch.randn(100),)
        edge_gm = (
            exir.capture(f, x, CaptureConfig(pt2_mode=True)).to_edge().graph_module
        )
        validation_f = add_assertions(edge_gm)

        # This should run successfully since the inputs are the same size
        validation_f(torch.randn(100))

        # A shape assertion within the model should fail
        with self.assertRaises(AssertionError):
            validation_f(torch.randn(2))

        # A type assertion within the model should fail
        with self.assertRaises(AssertionError):
            validation_f(torch.randn(100, dtype=torch.float64))

    def _test_trace_export(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.cos()

        traced_object = trace(f, (torch.ones(6),))
        self.assertTrue(isinstance(traced_object, Trace))

        # user will create this
        export_session = ExportSession(traced_object)
        with self.assertRaisesRegex(ExportError, "There are outstanding guards"):
            export_session.export()

        def test_rule(guard: Guard) -> Optional[GuardResolution]:
            if guard.guard_type == GuardType.TENSOR_MATCH:
                return GuardResolution.IGNORE
            return None

        export_session.add_guard_rule(test_rule)

        graph_module = export_session.export()
        assert graph_module is not None
        self.assertTrue(torch.equal(f(torch.ones(6)), graph_module(torch.ones(6))))

        def test_rule_strict(guard: Guard) -> Optional[GuardResolution]:
            if guard.guard_type == GuardType.TENSOR_MATCH:
                return GuardResolution.ERROR_AT_EXPORT
            return None

        export_session.add_guard_rule(test_rule_strict)
        self.assertEqual(len(export_session.guard_rules), 3)
        with self.assertRaisesRegex(ExportError, "There are outstanding guards"):
            export_session.export()

        export_session.add_guard_rule(test_rule)
        self.assertEqual(len(export_session.guard_rules), 4)

        graph_module = export_session.export()
        assert graph_module is not None
        self.assertTrue(torch.equal(f(torch.ones(6)), graph_module(torch.ones(6))))

    # [ExportErrorType.VIOLATION_OF_SPEC]: Cannot construct an input for graph module: GraphModule()
    @unittest.expectedFailure
    def test_pickle(self) -> None:
        import pickle

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        inputs = (torch.randn(1, 3), torch.randn(1, 3))

        gm_dynamo = exir.capture(f, inputs, exir.CaptureConfig(pt2_mode=True)).to_edge()
        pickled_gm = pickle.dumps(
            convert_fake_tensor_to_tensor_meta(copy.deepcopy(gm_dynamo))[0]
        )
        loaded_gm = pickle.loads(pickled_gm)
        self.assertTrue(torch.allclose(loaded_gm(*inputs)[0], gm_dynamo(*inputs)))

    # [ExportErrorType.VIOLATION_OF_SPEC]: Cannot construct an input for graph module: GraphModule()
    @unittest.expectedFailure
    def test_pickle_with_backend(self) -> None:
        import pickle

        m = models.CompositeDelegateModule()

        exec_prog = (
            exir.capture(m, m.get_random_inputs(), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .to_executorch()
        )
        graph_module = exec_prog.dump_graph_module()

        pickle_buf = pickle.dumps(
            convert_fake_tensor_to_tensor_meta(copy.deepcopy(graph_module))[0]
        )
        pickle.loads(pickle_buf)

    # [ExportErrorType.VIOLATION_OF_SPEC]: Cannot construct an input for graph module: GraphModule()
    @unittest.expectedFailure
    def test_pickle_meta(self) -> None:
        import pickle

        def f(x: torch.Tensor) -> torch.Tensor:
            def true_fn(x: torch.Tensor) -> torch.Tensor:
                def inner_true_fn(y: torch.Tensor) -> torch.Tensor:
                    return x + y

                return inner_true_fn(x)

            def false_fn(x: torch.Tensor) -> torch.Tensor:
                def inner_false_fn(y: torch.Tensor) -> torch.Tensor:
                    return x - y

                return inner_false_fn(x)

            return control_flow.cond(x.shape[0] < 10, true_fn, false_fn, [x])

        inputs = (torch.ones(3),)
        ep = exir.capture(f, inputs, exir.CaptureConfig(pt2_mode=True)).to_edge()

        # Pickle the ExportGraphModule
        pickled_ep = pickle.dumps(
            convert_fake_tensor_to_tensor_meta(copy.deepcopy(ep))[0]
        )
        loaded_ep = pickle.loads(pickled_ep)

        for node1, node2 in zip(loaded_ep.graph.nodes, ep.graph.nodes):
            val1 = node1.meta.get("val", None)
            val2 = node2.meta.get("val", None)

            if val1 is None or val2 is None:
                # Either both are None
                self.assertEqual(val1, val2)
            elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                # Or both are fake tensors with the same shape/dtype
                self.assertEqual(val1.shape, val2.shape)
                self.assertEqual(val1.dtype, val2.dtype)
            elif isinstance(val1, list) and isinstance(val2, list):
                # Or both are fake tensors lists with one element and with the
                # same shape/dtype
                self.assertTrue(len(val1) == len(val2) and len(val1) == 1)
                self.assertEqual(val1[0].shape, val2[0].shape)
                self.assertEqual(val1[0].dtype, val2[0].dtype)
            else:
                # For expressions like 's0 < 10' can only compare through string
                self.assertEqual(str(val1), str(val2))

        self.assertTrue(torch.allclose(loaded_ep(*inputs), ep(*inputs)))
