# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)


class TestQnnCompilerSpec(unittest.TestCase):
    def test_fcb_pairs_soc_and_htp_options(self):
        first = generate_htp_compiler_spec(use_fp16=False)
        second = generate_htp_compiler_spec(use_fp16=True)
        option = flatbuffer_to_option(
            generate_qnn_executorch_compiler_spec(
                soc_model=[QcomChipset.SM8650, QcomChipset.SM8750],
                backend_options=[first, second],
            )[0].value
        )

        self.assertTrue(option.fcb_options.fcb_reference_weight_sharing)
        self.assertEqual(
            [target.soc_info.soc_model for target in option.fcb_options.targets],
            [QcomChipset.SM8650, QcomChipset.SM8750],
        )
        self.assertEqual(
            [target.htp_options.precision for target in option.fcb_options.targets],
            [first.htp_options.precision, second.htp_options.precision],
        )

    def test_fcb_rejects_invalid_target_configuration(self):
        htp = generate_htp_compiler_spec(use_fp16=False)
        cases = [
            (
                {
                    "soc_model": [QcomChipset.SM8650],
                    "backend_options": [htp],
                },
                "at least two",
            ),
            (
                {
                    "soc_model": [QcomChipset.SM8650, QcomChipset.SM8750],
                    "backend_options": [htp],
                },
                "equal-length",
            ),
            (
                {
                    "soc_model": [QcomChipset.SM8650, QcomChipset.SM8750],
                    "backend_options": [htp, generate_gpu_compiler_spec()],
                },
                "HTP",
            ),
            (
                {
                    "soc_model": [QcomChipset.SM8650, QcomChipset.SM8750],
                    "backend_options": [htp, htp],
                    "online_prepare": True,
                },
                "offline_prepare",
            ),
            (
                {
                    "soc_model": QcomChipset.SM8650,
                    "backend_options": [htp, htp],
                },
                "lists",
            ),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, message
            ):
                generate_qnn_executorch_compiler_spec(**kwargs)

    def test_fcb_rejects_dlbc_when_reference_sharing_enabled(self):
        htp = generate_htp_compiler_spec(use_fp16=False, use_dlbc=True)
        with self.assertRaisesRegex(ValueError, "DLBC"):
            generate_qnn_executorch_compiler_spec(
                soc_model=[QcomChipset.SM8650, QcomChipset.SM8750],
                backend_options=[htp, htp],
            )

    def test_single_target_remains_non_fcb(self):
        option = flatbuffer_to_option(
            generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SM8650,
                backend_options=generate_htp_compiler_spec(use_fp16=False),
            )[0].value
        )
        self.assertIsNone(option.fcb_options)
        self.assertEqual(
            option.backend_options.backend_type,
            QnnExecuTorchBackendType.kHtpBackend,
        )


if __name__ == "__main__":
    unittest.main()
