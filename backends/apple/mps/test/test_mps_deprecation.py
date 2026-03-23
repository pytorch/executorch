#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import unittest
import warnings


class TestMPSDeprecation(unittest.TestCase):
    """Tests that MPS backend deprecation warnings are properly emitted."""

    def test_mps_package_import_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import executorch.backends.apple.mps  # noqa: F401

            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "deprecated" in str(warning.message).lower()
                and "mps" in str(warning.message).lower()
            ]
            self.assertTrue(
                len(future_warnings) > 0,
                "Importing executorch.backends.apple.mps should emit a FutureWarning",
            )

    def test_mps_partition_package_import_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import executorch.backends.apple.mps.partition  # noqa: F401

            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "deprecated" in str(warning.message).lower()
                and "mps" in str(warning.message).lower()
            ]
            self.assertTrue(
                len(future_warnings) > 0,
                "Importing executorch.backends.apple.mps.partition should emit a FutureWarning",
            )

    def test_mps_backend_class_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from executorch.backends.apple.mps.mps_preprocess import (  # noqa: F811
                MPSBackend,
            )

            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "MPS backend is deprecated" in str(warning.message)
            ]
            self.assertTrue(
                len(future_warnings) > 0,
                "Importing MPSBackend should emit a FutureWarning about deprecation",
            )

    def test_mps_partitioner_class_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner  # noqa: F811, F401

            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "MPS partitioner is deprecated" in str(warning.message)
            ]
            self.assertTrue(
                len(future_warnings) > 0,
                "Importing MPSPartitioner should emit a FutureWarning about deprecation",
            )

    def test_deprecation_message_mentions_alternative(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from executorch.backends.apple.mps.mps_preprocess import (  # noqa: F811
                MPSBackend,
            )

            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
            ]
            self.assertTrue(len(future_warnings) > 0)
            message = str(future_warnings[-1].message)
            self.assertIn(
                "CoreML",
                message,
                "Deprecation warning should mention CoreML as an alternative",
            )

    def test_deprecation_message_mentions_removal_version(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from executorch.backends.apple.mps.mps_preprocess import (  # noqa: F811
                MPSBackend,
            )

            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
            ]
            self.assertTrue(len(future_warnings) > 0)
            message = str(future_warnings[-1].message)
            self.assertIn(
                "1.5",
                message,
                "Deprecation warning should mention the removal version (1.5)",
            )
