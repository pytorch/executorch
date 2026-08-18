# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../production_debt.py",
)
spec = importlib.util.spec_from_file_location("executorch_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["executorch_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtProgramGate = production_debt_mod.ProductionDebtProgramGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtProgramGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtProgramGate(
            never_equate_intent_to_approval=True,
            max_acceptable_edi=12.0,
        )

    def test_clean_program_passes_readiness(self) -> None:
        report = self.gate.evaluate_compiled_program(
            program_id="llama3_1b_meta_quest_pte",
            allocated_static_arena_bytes=250000000,
            utilized_tensor_bytes=260000000,
            method_execution_latency_ms=18.5,
            delegate_fallback_nodes=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.edi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_program_fails_debt(self) -> None:
        report = self.gate.evaluate_compiled_program(
            program_id="uncalibrated_pte_program",
            allocated_static_arena_bytes=250000000,
            utilized_tensor_bytes=700000000,  # 2.8x static arena sprawl
            method_execution_latency_ms=110.0,  # High latency
            delegate_fallback_nodes=3,  # 3 delegate fallback nodes
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.edi_score, 50.0)
        self.assertIn("HIGH_STATIC_ARENA_SPRAWL_2.80X", report.critical_smells)
        self.assertIn("HIGH_METHOD_EXECUTION_LATENCY_110.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_DELEGATE_FALLBACK_NODES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_MUTATION_OPERATORS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_compiled_program("prog-1")
        self.gate.evaluate_compiled_program("prog-2")
        self.gate.evaluate_compiled_program("prog-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
