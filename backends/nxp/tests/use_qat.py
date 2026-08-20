# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest


@pytest.fixture
def use_qat(request):
    return request.param


def pytest_generate_tests(metafunc):
    if "use_qat" in metafunc.fixturenames:
        metafunc.parametrize(
            "use_qat",
            [True, False],
            indirect=True,
            ids=lambda use_qat: "QAT" if use_qat else "PTQ",
        )
