# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from executorch.exir.dialects.edge.spec.utils import SAMPLE_INPUT

# Add edge ops which we lower but which are not included in exir/dialects/edge/edge.yaml here.
CUSTOM_EDGE_OPS = [
    "linspace.default",
    "cond.default",
    "eye.default",
    "expm1.default",
    "vector_norm.default",
    "hardsigmoid.default",
    "hardswish.default",
    "linear.default",
    "maximum.default",
    "mean.default",
    "multihead_attention.default",
    "adaptive_avg_pool2d.default",
    "bitwise_right_shift.Tensor",
    "bitwise_right_shift.Scalar",
    "bitwise_left_shift.Tensor",
    "bitwise_left_shift.Scalar",
    "native_group_norm.default",
    "silu.default",
    "sdpa.default",
    "sum.default",
    "unbind.int",
    "unflatten.int",
    "_native_batch_norm_legit_no_training.default",
    "_native_batch_norm_legit.no_stats",
    "alias_copy.default",
    "pixel_shuffle.default",
    "pixel_unshuffle.default",
    "while_loop.default",
]
ALL_EDGE_OPS = SAMPLE_INPUT.keys() | CUSTOM_EDGE_OPS

# Add all targets and TOSA profiles we support here.
TARGETS = [
    "tosa_FP",
    "tosa_INT",
    "u55_INT",
    "u85_INT",
    "vgf_INT",
    "vgf_FP",
    "vgf_quant",
    "vgf_no_quant",
]


def get_op_name_map():
    """
    Returns a mapping from names on the form to be used in unittests to edge op:
    1. Names are in lowercase.
    2. Overload is ignored if 'default', otherwise it's appended with an underscore.
    3. Overly verbose name are shortened by removing certain prefixes/suffixes.

    Examples:
        abs.default -> abs
        split_copy.Tensor -> split_tensor
    """
    map = {}
    for edge_name in ALL_EDGE_OPS:
        op, overload = edge_name.split(".")

        # Normalize names
        op = op.lower()
        op = op.removeprefix("_")
        op = op.removesuffix("_copy")
        op = op.removesuffix("_with_indices")
        overload = overload.lower()

        if overload == "default":
            map[op] = edge_name
        else:
            map[f"{op}_{overload}"] = edge_name

    return map


def parse_test_name(
    test_name: str, op_name_map: dict[str, str]
) -> tuple[str, str, bool]:
    """
    Parses a test name on the form
        test_OP_TARGET_<not_delegated>_<any_other_info>
    where OP must match a key in op_name_map and TARGET one string in TARGETS. The
    "not_delegated" suffix indicates that the test tests that the op is not delegated.

    Examples of valid names: "test_mm_u55_INT_not_delegated" and
    "test_add_scalar_tosa_FP_two_inputs".

    Returns a tuple (OP, TARGET, IS_DELEGATED) if valid.
    """
    test_name = test_name.removeprefix("test_")
    is_delegated = "not_delegated" not in test_name
    assert (
        "reject" not in test_name
    ), f"Use 'not_delegated' instead of 'reject' in {test_name}"

    op = "None"
    target = "None"
    for potential_target in TARGETS:
        index = test_name.find(potential_target)
        if index != -1:
            op = test_name[: index - 1]
            target = potential_target
            break
    # Special case for convolution
    op = op.removesuffix("_1d")
    op = op.removesuffix("_2d")
    op = op.removesuffix("_3d")

    # Remove suffix for 16 bit activation and 8 bit weight test cases
    op = op.removesuffix("_16a8w")

    assert target != "None", f"{test_name} does not contain one of {TARGETS}"
    assert (
        op in op_name_map.keys()
    ), f"Parsed unvalid OP from {test_name}, {op} does not exist in edge.yaml or CUSTOM_EDGE_OPS"

    return op, target, is_delegated


if __name__ == "__main__":
    """Parses a list of test names given on the commandline."""
    import sys

    sys.tracebacklimit = 0  # Do not print stack trace

    op_name_map = get_op_name_map()
    exit_code = 0

    for test_name in sys.argv[1:]:
        try:
            assert test_name[:5] == "test_", f"Unexpected input: {test_name}"
            parse_test_name(test_name, op_name_map)
        except AssertionError as e:
            print(e)
            exit_code = 1
        else:
            print(f"{test_name} OK")

    sys.exit(exit_code)
