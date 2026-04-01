# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import difflib
import logging

from executorch.backends.arm.scripts.testname_rules.collect_testname_resources import (
    MODEL_LIST,
    OP_LIST,
    PASS_LIST,
    TARGETS,
)


logger = logging.getLogger(__name__)


class TestNameViolation:
    def __init__(self, test_name: str, message: str):
        self.test_name = test_name
        self.message = message

    def __str__(self) -> str:
        msg_indented = "\n".join("    " + line for line in self.message.splitlines())
        return f"Invalid test name for {self.test_name}\n{msg_indented}\n"

    def __repr__(self) -> str:
        return self.__str__()


def _match_allowed_op_prefix(
    test_name: str,
) -> tuple[str | None, str | None, bool, bool]:
    test_name = test_name.removeprefix("test_")
    is_delegated = "not_delegated" not in test_name
    is_16x8_quantized = False

    op = None
    target = None
    for potential_target in TARGETS:
        index = test_name.find(potential_target)
        if index != -1:
            op = test_name[: index - 1]
            target = potential_target
            if ("16a8w" in test_name) or ("a16w8" in test_name):
                is_16x8_quantized = True
            break

    if op is not None:
        # Special case for convolution
        op = op.removesuffix("_1d")
        op = op.removesuffix("_2d")
        op = op.removesuffix("_3d")

        # Remove suffix for 16 bit activation and 8 bit weight test cases
        op = op.removesuffix("_16a8w")

    return op, target, is_16x8_quantized, is_delegated


def _match_allowed_model_prefix(token: str, allowed_models: list[str]) -> str | None:
    for allowed_model in allowed_models:
        if token == allowed_model:
            return allowed_model
    return None


def _match_allowed_pass_prefix(token: str, allowed_passes: list[str]) -> str | None:
    for allowed_pass in allowed_passes:
        if token == allowed_pass:
            return allowed_pass
    return None


def _extract_target(name: str) -> str | None:
    # The target is the last supported target token in the name, optionally
    # followed by extra suffix data such as "_not_delegated".
    for target in TARGETS:
        marker = f"_{target}"
        idx = name.rfind(marker)
        if idx == -1:
            continue

        suffix_idx = idx + len(marker)
        if suffix_idx == len(name) or name[suffix_idx] == "_":
            return target
    return None


def _parse_test_name_tokens(name: str) -> tuple[str, str | None]:
    rest = name[5:]
    target = _extract_target(name)
    token = rest
    if target:
        idx = rest.rfind(target)
        token = rest[:idx].rstrip("_")

    return token, target


def _get_parsing_info(kind: str, name: str) -> str:
    token, target = _parse_test_name_tokens(name)

    return f"{kind} token parsed as '{token}'\n" f"TARGET token parsed as '{target}'"


def parse_op_test(test_name: str) -> tuple[str, str, bool, bool] | TestNameViolation:
    matched_op, target, quantized_16x8, delegated = _match_allowed_op_prefix(test_name)

    if "reject" in test_name:
        return TestNameViolation(
            test_name,
            "Use 'not_delegated' instead of 'reject' in test names",
        )

    if not matched_op:
        parsing_info = _get_parsing_info("OP", test_name)
        return TestNameViolation(
            test_name,
            (
                f"Expected test_OP_TARGET_*\n"
                f"OP token not found or invalid\n"
                f"{parsing_info}"
            ),
        )

    if target is None:
        parsing_info = _get_parsing_info("OP", test_name)
        return TestNameViolation(
            test_name,
            (
                f"Expected test_OP_TARGET_*\n"
                f"TARGET is None (valid targets: {TARGETS}))\n"
                f"{parsing_info}"
            ),
        )

    if matched_op not in OP_LIST:
        parsing_info = _get_parsing_info("OP", test_name)
        closest_match = difflib.get_close_matches(matched_op, OP_LIST, n=1, cutoff=0.0)[
            0
        ]
        return TestNameViolation(
            test_name,
            (
                f"Expected test_OP_TARGET_*\n"
                f"OP '{matched_op}' not recognized (closest match: {closest_match})\n"
                f"{parsing_info}"
            ),
        )

    result = (matched_op, target, quantized_16x8, delegated)
    logger.debug('Parsed op test "%s": %s', test_name, result)

    return result


def parse_model_test(test_name: str) -> tuple[str, str] | TestNameViolation:
    token, target = _parse_test_name_tokens(test_name)

    if not token:
        parsing_info = _get_parsing_info("MODEL", test_name)
        return TestNameViolation(
            test_name,
            (
                f"Expected test_MODEL_TARGET_*\n"
                f"MODEL token not found or invalid\n"
                f"{parsing_info}\n"
            ),
        )

    if not target:
        parsing_info = _get_parsing_info("MODEL", test_name)
        return TestNameViolation(
            test_name,
            (
                f"Expected test_MODEL_TARGET_*\n"
                f"TARGET token not found (valid targets: {TARGETS})\n"
                f"{parsing_info}"
            ),
        )

    matched_model = _match_allowed_model_prefix(token, MODEL_LIST)
    if matched_model is None:
        parsing_info = _get_parsing_info("MODEL", test_name)
        closest_match = difflib.get_close_matches(token, MODEL_LIST, n=1, cutoff=0.0)[0]
        return TestNameViolation(
            test_name,
            (
                f"Expected test_MODEL_TARGET_*\n"
                f"MODEL {token} not recognized (closest match: {closest_match})\n"
                f"{parsing_info}"
            ),
        )

    result = (token, target)
    logger.debug('Parsed model test "%s": %s', test_name, result)

    return result


def parse_pass_test(test_name: str) -> tuple[str, str] | TestNameViolation:
    pass_, target = _parse_test_name_tokens(test_name)

    if not pass_:
        parsing_info = _get_parsing_info("PASS", test_name)
        return TestNameViolation(
            test_name,
            (
                f"Expected test_PASS_TARGET_*\n"
                f"PASS token not found or invalid\n"
                f"{parsing_info}\n"
            ),
        )

    if not target:
        parsing_info = _get_parsing_info("PASS", test_name)
        return TestNameViolation(
            test_name,
            (
                f"Expected test_PASS_TARGET_*\n"
                f"TARGET token not found (valid targets: {TARGETS})\n"
                f"{parsing_info}"
            ),
        )

    matched_pass = _match_allowed_pass_prefix(pass_, PASS_LIST)
    if matched_pass is None:
        parsing_info = _get_parsing_info("PASS", test_name)
        closest_match = difflib.get_close_matches(pass_, PASS_LIST, n=1, cutoff=0.0)[0]
        return TestNameViolation(
            test_name,
            (
                f"Expected test_PASS_TARGET_* with PASS in PASS_LIST\n"
                f"PASS '{pass_} not recognized (closest match: {closest_match})'\n"
                f"{parsing_info}"
            ),
        )

    result = (pass_, target)
    logger.debug('Parsed pass test "%s": %s', test_name, result)
    return result


def parse_general_test(test_name: str) -> tuple[str, str] | TestNameViolation:
    name, target = _parse_test_name_tokens(test_name)

    if not name:
        parsing_info = _get_parsing_info("NAME", test_name)
        return TestNameViolation(
            test_name,
            f"Expected test_*_TARGET_*\n" "Invalid NAME token\n" f"{parsing_info}",
        )

    if not target:
        parsing_info = _get_parsing_info("NAME", test_name)
        return TestNameViolation(
            test_name,
            (
                "Expected test_*_TARGET_*\n"
                f"TARGET token not found (valid targets: {TARGETS})\n"
                f"{parsing_info}"
            ),
        )

    result = (name, target)
    logger.debug('Parsed general test "%s": %s', test_name, result)
    return result
