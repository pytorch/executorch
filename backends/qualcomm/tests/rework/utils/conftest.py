# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from executorch.backends.qualcomm.export_utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    make_quantizer,
    QcomChipset,
)
from executorch.backends.qualcomm.tests.rework.conftest import default_property


@pytest.fixture(scope="session")
def compile_spec():
    return generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, default_property().soc_model),
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )


@pytest.fixture(scope="session")
def quantizer():
    return make_quantizer(soc_model=default_property().soc_model)
