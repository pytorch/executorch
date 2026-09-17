# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock, patch

from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils import qnn_manager_lifecycle as lifecycle


class TestQnnManagerDeviceCache(unittest.TestCase):
    def test_registry_keys_managers_by_backend_and_soc(self):
        managers = [Mock(), Mock(), Mock()]
        for manager in managers:
            manager.InitBackend.return_value = Mock(value=0)

        with (
            patch.object(lifecycle, "setup_qnn_sdk"),
            patch.object(lifecycle, "disable_mkldnn_on_amd"),
            patch.object(
                lifecycle.PyQnnManager, "QnnManager", side_effect=managers
            ) as create,
        ):
            registry = lifecycle.QnnManagerRegistry()
            first = registry.get_or_create_qnn_manager(
                QnnExecuTorchBackendType.kHtpBackend,
                b"first",
                QcomChipset.SM8650,
            )
            second = registry.get_or_create_qnn_manager(
                QnnExecuTorchBackendType.kHtpBackend,
                b"second",
                QcomChipset.SM8750,
            )
            third = registry.get_or_create_qnn_manager(
                QnnExecuTorchBackendType.kHtpBackend,
                b"third",
                QcomChipset.SM8650,
            )

        self.assertEqual(create.call_count, 2)
        self.assertIsNot(first, second)
        self.assertIs(first, third)


if __name__ == "__main__":
    unittest.main()
