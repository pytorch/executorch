# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.aten_passes.convert_scalar_to_attr import (
    ConvertScalarToAttrPass,
)
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import neutron_target_spec
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import (
    AddScalarModule,
    MulScalarModule,
    SubScalarModule,
)
from executorch.backends.nxp.tests.nsys_testing import AllCloseOutputComparator, lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import AddTensor, MulTensor, SubTensor


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class TestConvertScalarToAttr:
    @pytest.mark.parametrize(
        "model_cls, expected_op",
        [
            pytest.param(AddScalarModule, torch.ops.aten.add.Tensor, id="Add op."),
            pytest.param(SubScalarModule, torch.ops.aten.sub.Tensor, id="Sub op."),
            pytest.param(MulScalarModule, torch.ops.aten.mul.Tensor, id="Mul op."),
        ],
    )
    def test__scalar_to_attr(self, model_cls, expected_op):
        input_shape = (2, 5, 7, 9)
        model = model_cls(scalar=2.0)

        example_input = torch.rand(input_shape)
        exir_program_aten = torch.export.export(model, (example_input,)).module()

        # Check if the node with scalar arg did not disappear.
        assert graph_contains_any_of_ops(exir_program_aten.graph, [expected_op])
        outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

        # Apply the optimization.
        NeutronAtenPassManager(
            neutron_target_spec, [ConvertScalarToAttrPass()]
        )(exir_program_aten)

        exp_op_node = [
            n for n in exir_program_aten.graph.nodes if n.target == expected_op
        ][0]
        # Check that no arg is `float` or `int`.
        # Note: `bool` is subtype of `int`, but `bool` does not need to be converted to `get_attr`.
        assert not any(
            isinstance(arg, (int, float)) and not isinstance(arg, bool) for arg in exp_op_node.args
        )

        outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

        # Make sure the model still produces the exact same output.
        assert len(outputs_before) == len(outputs_after)
        for i in range(len(outputs_before)):
            assert np.allclose(outputs_before[i], outputs_after[i])

    @pytest.mark.parametrize(
        "model_cls, expected_op",
        [
            pytest.param(AddScalarModule, AddTensor, id="Add op."),
            pytest.param(SubScalarModule, SubTensor, id="Sub op."),
            pytest.param(MulScalarModule, MulTensor, id="Mul op."),
        ],
    )
    def test__scalar_to_attr__full_pipeline(
        self, mocker, request, model_cls, expected_op,
    ):
        input_shape = (2, 7, 5, 11)
        model = model_cls(scalar=2.0)

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={expected_op: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            remove_quant_io_ops=remove_quant_io_ops,
        )

