# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import sys
from pathlib import Path

from executorch.backends.arm.scripts.testname_rules.collect_tests import (
    collect_test_files,
    collect_tests,
)

from executorch.backends.arm.scripts.testname_rules.parse_test_names import (
    parse_general_test,
    parse_model_test,
    parse_op_test,
    parse_pass_test,
    TestNameViolation,
)

LOGGER = logging.getLogger(__name__)


def _is_in_path(child: str, parent: str) -> bool:
    """Returns True if 'child' path is inside 'parent' path."""
    child = Path(child).resolve()
    parent = Path(parent).resolve()
    return parent == child or parent in child.parents


def check_test_name_validations(
    tests: list[str],
) -> list[TestNameViolation]:
    violations: list[TestNameViolation] = []
    for test in tests:
        path, test_name = test.split("::")
        result: tuple[str, str, bool, bool] | tuple[str, str] | TestNameViolation | None

        if _is_in_path(path, "backends/arm/test/ops"):
            result = parse_op_test(test_name)
        elif _is_in_path(path, "backends/arm/test/models"):
            result = parse_model_test(test_name)
        elif _is_in_path(path, "backends/arm/test/passes"):
            result = parse_pass_test(test_name)
        elif _is_in_path(path, "backends/arm/test/quantizer") or _is_in_path(
            path, "backends/arm/test/misc"
        ):
            result = parse_general_test(test_name)
        else:
            result = None

        if isinstance(result, TestNameViolation):
            violations.append(result)

    return violations


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tests: list[str] = []
    path_list: list[Path] = [Path(path) for path in sys.argv[1:]]

    if path_list == []:
        tests = collect_tests(collect_test_files(Path("backends/arm/test")))
    else:
        tests = collect_tests(path_list)

    violations = check_test_name_validations(tests)

    for entry in violations:
        LOGGER.error("%s", entry)

    LOGGER.info("Total tests needing renaming: %d", len(violations))

    if violations:
        LOGGER.info(
            "Please follow the test naming convention: https://confluence.arm.com/display/MLENG/Executorch+naming+conventions"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
