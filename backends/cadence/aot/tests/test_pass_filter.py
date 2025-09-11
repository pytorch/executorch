# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from copy import deepcopy

from typing import Callable, Type

from executorch.backends.cadence.aot import pass_utils
from executorch.backends.cadence.aot.pass_utils import (
    ALL_CADENCE_PASSES,
    CadencePassAttribute,
    create_cadence_pass_filter,
    register_cadence_pass,
)

from executorch.exir.pass_base import ExportPass, PassBase


class TestBase(unittest.TestCase):
    def setUp(self) -> None:
        # Before running each test, create a copy of _all_passes to later restore it after test.
        # This avoids messing up the original _all_passes when running tests.
        self._all_passes_original = deepcopy(ALL_CADENCE_PASSES)
        # Clear _all_passes to do a clean test. It'll be restored after each test in tearDown().
        pass_utils.ALL_CADENCE_PASSES.clear()

    def tearDown(self) -> None:
        # Restore _all_passes to original state before test.
        pass_utils.ALL_CADENCE_PASSES = self._all_passes_original

    def get_filtered_passes(
        self, filter_: Callable[[Type[PassBase]], bool]
    ) -> dict[Type[PassBase], CadencePassAttribute]:
        return {c: attr for c, attr in ALL_CADENCE_PASSES.items() if filter_(c)}


# Test pass registration
class TestPassRegistration(TestBase):
    def test_register_cadence_pass(self) -> None:
        pass_attr_O0 = CadencePassAttribute(opt_level=0)
        pass_attr_debug = CadencePassAttribute(opt_level=None, debug_pass=True)
        pass_attr_O1_all_backends = CadencePassAttribute(
            opt_level=1,
        )

        # Register 1st pass with opt_level=0
        @register_cadence_pass(pass_attr_O0)
        class DummyPass_O0(ExportPass):
            pass

        # Register 2nd pass with opt_level=1, all backends.
        @register_cadence_pass(pass_attr_O1_all_backends)
        class DummyPass_O1_All_Backends(ExportPass):
            pass

        # Register 3rd pass with opt_level=None, debug=True
        @register_cadence_pass(pass_attr_debug)
        class DummyPass_Debug(ExportPass):
            pass

        # Check if the three passes are indeed added into _all_passes
        expected_all_passes = {
            DummyPass_O0: pass_attr_O0,
            DummyPass_Debug: pass_attr_debug,
            DummyPass_O1_All_Backends: pass_attr_O1_all_backends,
        }
        self.assertEqual(pass_utils.ALL_CADENCE_PASSES, expected_all_passes)


# Test pass filtering
class TestPassFiltering(TestBase):
    def test_filter_none(self) -> None:
        pass_attr_O0 = CadencePassAttribute(opt_level=0)
        pass_attr_O1_debug = CadencePassAttribute(opt_level=1, debug_pass=True)
        pass_attr_O1_all_backends = CadencePassAttribute(
            opt_level=1,
        )

        @register_cadence_pass(pass_attr_O0)
        class DummyPass_O0(ExportPass):
            pass

        @register_cadence_pass(pass_attr_O1_debug)
        class DummyPass_O1_Debug(ExportPass):
            pass

        @register_cadence_pass(pass_attr_O1_all_backends)
        class DummyPass_O1_All_Backends(ExportPass):
            pass

        O1_filter = create_cadence_pass_filter(opt_level=1, debug=True)
        O1_filter_passes = self.get_filtered_passes(O1_filter)

        # Assert that no passes are filtered out.
        expected_passes = {
            DummyPass_O0: pass_attr_O0,
            DummyPass_O1_Debug: pass_attr_O1_debug,
            DummyPass_O1_All_Backends: pass_attr_O1_all_backends,
        }
        self.assertEqual(O1_filter_passes, expected_passes)

    def test_filter_debug(self) -> None:
        pass_attr_O1_debug = CadencePassAttribute(opt_level=1, debug_pass=True)
        pass_attr_O2 = CadencePassAttribute(opt_level=2)

        @register_cadence_pass(pass_attr_O1_debug)
        class DummyPass_O1_Debug(ExportPass):
            pass

        @register_cadence_pass(pass_attr_O2)
        class DummyPass_O2(ExportPass):
            pass

        debug_filter = create_cadence_pass_filter(opt_level=2, debug=False)
        debug_filter_passes = self.get_filtered_passes(debug_filter)

        # Assert that debug passees are filtered out, since the filter explicitly
        # chooses debug=False.
        self.assertEqual(debug_filter_passes, {DummyPass_O2: pass_attr_O2})

    def test_filter_all(self) -> None:
        @register_cadence_pass(CadencePassAttribute(opt_level=1))
        class DummyPass_O1(ExportPass):
            pass

        @register_cadence_pass(CadencePassAttribute(opt_level=2))
        class DummyPass_O2(ExportPass):
            pass

        debug_filter = create_cadence_pass_filter(opt_level=0)
        debug_filter_passes = self.get_filtered_passes(debug_filter)

        # Assert that all the passes are filtered out, since the filter only selects
        # passes with opt_level <= 0
        self.assertEqual(debug_filter_passes, {})

    def test_filter_opt_level_None(self) -> None:
        pass_attr_O1 = CadencePassAttribute(opt_level=1)
        pass_attr_O2_debug = CadencePassAttribute(opt_level=2, debug_pass=True)

        @register_cadence_pass(CadencePassAttribute(opt_level=None))
        class DummyPass_None(ExportPass):
            pass

        @register_cadence_pass(pass_attr_O1)
        class DummyPass_O1(ExportPass):
            pass

        @register_cadence_pass(pass_attr_O2_debug)
        class DummyPass_O2_Debug(ExportPass):
            pass

        O2_filter = create_cadence_pass_filter(opt_level=2, debug=True)
        filtered_passes = self.get_filtered_passes(O2_filter)
        # Passes with opt_level=None should never be retained.
        expected_passes = {
            DummyPass_O1: pass_attr_O1,
            DummyPass_O2_Debug: pass_attr_O2_debug,
        }
        self.assertEqual(filtered_passes, expected_passes)
