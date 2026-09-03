# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import unittest

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig

from executorch.examples.samsung.scripts.mobilebert_finetune import MobileBertFinetune
from transformers import AutoTokenizer


def patch_mobilebert_finetuning():
    def _monkeypatch_load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            do_lower_case=True,
        )
        return tokenizer

    old_func = MobileBertFinetune.load_tokenizer
    MobileBertFinetune.load_tokenizer = _monkeypatch_load_tokenizer
    return old_func


def recover_mobilebert_finetuning(old_func):
    MobileBertFinetune.load_tokenizer = old_func


class Test_Milestone_MobileBertFinetune(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_func = patch_mobilebert_finetuning(cls.model_cache_dir)

    @classmethod
    def tearDownClass(cls):
        recover_mobilebert_finetuning(cls._old_func)

    def test_mobilebert_finetuning_fp16(self):
        mobilebert_finetune = MobileBertFinetune()
        model, _ = mobilebert_finetune.get_finetune_mobilebert(None)
        example_input = mobilebert_finetune.get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )

        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.008)
        )
