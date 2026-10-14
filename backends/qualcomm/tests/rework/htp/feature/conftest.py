# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from functools import lru_cache

import pytest

from executorch.backends.qualcomm.export_utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)


@pytest.fixture(scope="session")
def compile_specs():
    @lru_cache()
    def _build(kwargs_config):
        kwargs = dict(kwargs_config)
        et_compile_spec_sig = set(
            inspect.signature(generate_qnn_executorch_compiler_spec).parameters.keys()
        )
        et_compile_spec_kwargs = {
            k: kwargs[k] for k in kwargs.keys() if k in et_compile_spec_sig
        }
        for k in et_compile_spec_kwargs.keys():
            kwargs.pop(k)

        return generate_qnn_executorch_compiler_spec(
            backend_options=generate_htp_compiler_spec(**kwargs),
            **et_compile_spec_kwargs,
        )

    return lambda kwargs_config: _build(kwargs_config)


@pytest.fixture(scope="session")
def fcb_compile_specs():
    @lru_cache()
    def _build(soc_models, fcb_reference_weight_sharing=True):
        return generate_qnn_executorch_compiler_spec(
            soc_model=list(soc_models),
            backend_options=[
                generate_htp_compiler_spec(
                    use_fp16=False,
                    use_weight_sharing=False,
                )
                for _ in soc_models
            ],
            fcb_reference_weight_sharing=fcb_reference_weight_sharing,
        )

    return _build
