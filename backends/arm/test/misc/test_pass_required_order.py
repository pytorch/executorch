# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List, Set, Type

import pytest
from executorch.backends.arm._passes.arm_pass_manager import ArmPass, ArmPassManager
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.pass_base import ExportPass


class PassC(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = set()


class PassB(ArmPass):
    _passes_required_after = {PassC}


class PassA(ArmPass):
    _passes_required_after = {PassB, PassC}


class IndependentPass(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = set()


def _setup_pass_manager(passes: List[ArmPass] | None = None):
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.00+INT")
    pass_manager = ArmPassManager(tosa_spec)
    if passes is not None:
        for p in passes:
            pass_manager.add_pass(p)
    return pass_manager


def test_no_passes():
    pass_manager = _setup_pass_manager()
    pass_manager.validate_constraints_mandatory()


def test_correct_order():
    pass_manager = _setup_pass_manager([PassA(), PassB(), PassC()])
    pass_manager.validate_constraints_mandatory()


def test_run_pass_twice():
    pass_manager = _setup_pass_manager([PassA(), PassB(), PassB(), PassC()])
    pass_manager.validate_constraints_mandatory()


def test_independent_pass():
    pass_manager = _setup_pass_manager(
        [
            IndependentPass(),
            PassA(),
            IndependentPass(),
            PassB(),
            IndependentPass(),
            PassC(),
            IndependentPass(),
        ]
    )
    pass_manager.validate_constraints_mandatory()


def test_duplicated_requiring_pass_put_last():
    error_msg = """The following constraints for passes are not met:
  - PassC must run after PassB
"""
    pass_manager = _setup_pass_manager([PassA(), PassB(), PassC(), PassB()])
    with pytest.raises(RuntimeError, match=re.escape(error_msg)):
        pass_manager.validate_constraints_mandatory()


def test_two_passes_wrong_order():
    error_msg = """The following constraints for passes are not met:
  - PassC must run after PassB
"""
    pass_manager = _setup_pass_manager([PassC(), PassB()])
    with pytest.raises(RuntimeError, match=re.escape(error_msg)):
        pass_manager.validate_constraints_mandatory()


def test_missing_passes():
    error_msg = """The following constraints for passes are not met:
  - PassC must run after PassA
  - PassC must run after PassB
"""
    pass_manager = _setup_pass_manager([PassA(), PassB()])
    with pytest.raises(RuntimeError, match=re.escape(error_msg)):
        pass_manager.validate_constraints_mandatory()
