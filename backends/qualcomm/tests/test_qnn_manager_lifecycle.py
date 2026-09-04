# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock, patch

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils import qnn_manager_lifecycle as lifecycle
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)


class TestQnnManagerLifecycle(unittest.TestCase):
    def setUp(self):
        lifecycle._current_qnn_managers.active_registry = None
        self.addCleanup(
            setattr, lifecycle._current_qnn_managers, "active_registry", None
        )

    def test_lazy_fcb_lookup_creates_each_target_once(self):
        specs = generate_qnn_executorch_compiler_spec(
            soc_model=[QcomChipset.SM8650, QcomChipset.SM8750],
            backend_options=[
                generate_htp_compiler_spec(use_fp16=False),
                generate_htp_compiler_spec(use_fp16=True),
            ],
        )
        managers = [Mock(), Mock()]
        for manager in managers:
            manager.InitBackend.return_value = Mock(value=0)

        with patch.object(
            lifecycle.PyQnnManager, "QnnManager", side_effect=managers
        ) as create:
            first = lifecycle.get_current_qnn_manager(specs, QcomChipset.SM8650)
            second = lifecycle.get_current_qnn_manager(specs, QcomChipset.SM8750)
            self.assertIs(
                first, lifecycle.get_current_qnn_manager(specs, QcomChipset.SM8650)
            )

        self.assertEqual(create.call_count, 2)
        self.assertIs(first, managers[0])
        self.assertIs(second, managers[1])

    def test_fcb_lookup_rejects_unknown_target(self):
        specs = generate_qnn_executorch_compiler_spec(
            soc_model=[QcomChipset.SM8650, QcomChipset.SM8750],
            backend_options=[
                generate_htp_compiler_spec(use_fp16=False),
                generate_htp_compiler_spec(use_fp16=False),
            ],
        )
        with self.assertRaisesRegex(ValueError, "do not target SM8850"):
            lifecycle.get_current_qnn_manager(specs, QcomChipset.SM8850)


if __name__ == "__main__":
    unittest.main()
