# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import pytest

from executorch.backends.qualcomm.tests.rework.conftest import Tolerance
from executorch.backends.qualcomm.tests.rework.src.feature import *  # noqa: F403


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": nullcontext()}, id="e2e"),
    ],
)
def test_logging(request, kwargs):
    Logging.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_multi_graph_weight_sharing(request, kwargs):
    MultiGraph.test_weight_sharing(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_multi_graph_inference(request, kwargs):
    MultiGraph.test_inference(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_online_prepare(request, kwargs):
    OnlinePrepare.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": nullcontext()}, id="e2e"),
    ],
)
def test_performance(request, kwargs):
    Performance.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": nullcontext()}, id="e2e"),
    ],
)
def test_profile(request, kwargs):
    Profile.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": nullcontext()}, id="e2e"),
    ],
)
def test_saver(request, kwargs):
    Saver.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_shared_buffer(request, kwargs):
    SharedBuffer.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": nullcontext()}, id="e2e"),
    ],
)
def test_spill_fill(request, kwargs):
    SpillFill.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": nullcontext()}, id="e2e"),
    ],
)
def test_tensor_dump(request, kwargs):
    TensorDump.test(request, kwargs)  # noqa: F405


def test_fcb_compiler_spec_preserves_targets(fcb_compile_specs):
    Fcb.compiler_spec_preserves_targets(fcb_compile_specs)  # noqa: F405


def test_fcb_manager_cache_is_keyed_by_soc(monkeypatch):
    Fcb.manager_cache_is_keyed_by_soc(monkeypatch)  # noqa: F405


def test_fcb_dlc_handle_enforces_lifetime_and_owner(fcb_compile_specs):
    Fcb.dlc_handle_enforces_lifetime_and_owner(fcb_compile_specs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_fcb_e2e(request, kwargs):
    Fcb.e2e(request, kwargs)  # noqa: F405


def test_fcb_reference_weight_sharing_reduces_pte_size(request):
    Fcb.reference_weight_sharing_reduces_pte_size(request, {})  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_fcb_reference_weight_sharing_e2e(request, kwargs):
    Fcb.reference_weight_sharing_e2e(request, kwargs)  # noqa: F405
