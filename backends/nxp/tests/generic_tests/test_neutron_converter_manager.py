# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import pickle

from executorch.backends.nxp.backend.neutron_converter_manager import (
    NeutronConverterManager,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import LinearModule


def test_conv2d_neutron_conversion__prefetching(mocker):
    model = LinearModule(True)
    input_shape = (1, 1, 32, 32)

    converter_spy = mocker.spy(NeutronConverterManager, "convert")
    _ = to_quantized_edge_program(
        model, input_shape, fetch_constants_to_sram=True
    ).exported_program()
    neutron_model_prefetch = converter_spy.spy_return

    _ = to_quantized_edge_program(
        model, input_shape, fetch_constants_to_sram=False
    ).exported_program()
    neutron_model_regular = converter_spy.spy_return

    assert len(neutron_model_prefetch) != len(
        neutron_model_regular
    ), "The weight prefetching flag does not make a difference!"


def test_convert_unsafe_args_are_picklable(mocker):
    """Verify that all args passed to `multiprocessing.Process` are picklable.

    The subprocess uses forkserver/spawn in some environments, which requires
    all Process args to be serializable via pickle.
    """
    model = LinearModule(True)
    input_shape = (1, 1, 32, 32)

    process_spy = mocker.spy(multiprocessing, "Process")
    to_quantized_edge_program(model, input_shape).exported_program()

    args = process_spy.call_args.kwargs["args"]
    for i, arg in enumerate(args):
        try:
            pickle.dumps(arg)
        except (pickle.PicklingError, TypeError) as e:
            raise AssertionError(
                f"Process arg at index {i} ({type(arg).__name__}) is not picklable: {e}"
            )
