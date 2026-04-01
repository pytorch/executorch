# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import logging
import os
import unittest

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.examples.models.inception_v4 import InceptionV4Model


def patch_iv4(weight_path: str):
    assert os.path.isfile(weight_path), "Can not found weight path for iv4"
    from safetensors import safe_open
    from timm.models import inception_v4

    def _monkeypatch_get_eager_model(self):
        tensors = {}
        with safe_open(weight_path, framework="pt") as st:
            for k in st.keys():
                tensors[k] = st.get_tensor(k)
        logging.info("Loading inception_v4 model")
        m = inception_v4(pretrained=True, pretrained_cfg={"state_dict": tensors})
        logging.info("Loaded inception_v4 model")
        return m

    old_func = InceptionV4Model.get_eager_model
    InceptionV4Model.get_eager_model = _monkeypatch_get_eager_model
    return old_func


def recover_iv4(old_func):
    InceptionV4Model.get_eager_model = old_func


class TestMilestoneInceptionV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert (model_cache_dir := os.getenv("MODEL_CACHE")), "MODEL_CACHE not set!"
        weight_path = os.path.join(
            model_cache_dir, os.path.join(model_cache_dir, "iv4/model.safetensors")
        )
        cls._old_func = patch_iv4(weight_path)

    @classmethod
    def tearDownClass(cls):
        recover_iv4(cls._old_func)

    def test_inception_v4_fp16(self):
        model = InceptionV4Model().get_eager_model()
        example_input = InceptionV4Model().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.02, rtol=0.02)
        )
