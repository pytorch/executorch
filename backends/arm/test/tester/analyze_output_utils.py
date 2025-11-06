# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import tempfile
from typing import Any, cast, Sequence

import torch
from executorch.backends.arm.test.runner_utils import (
    get_input_quantization_params,
    get_output_quantization_params,
)

from executorch.backends.test.harness.stages import StageType

logger = logging.getLogger(__name__)


TensorLike = torch.Tensor | tuple[torch.Tensor, ...]


def _ensure_tensor(value: TensorLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("Expected a Tensor or a non-empty tuple of Tensors")


def _print_channels(
    result: torch.Tensor,
    reference: torch.Tensor,
    channels_close: Sequence[bool],
    C: int,
    H: int,
    W: int,
    rtol: float,
    atol: float,
) -> str:

    output_str = ""
    exp = "000"
    booldata = False
    if reference.dtype == torch.bool or result.dtype == torch.bool:
        booldata = True

    for c in range(C):
        if channels_close[c]:
            continue
        if not booldata:
            max_diff = torch.max(torch.abs(reference - result))
            exp = f"{max_diff:2e}"[-3:]
            output_str += f"channel {c} (e{exp})\n"
        else:
            max_diff = torch.max(reference ^ result)
            output_str += f"channel {c} (bool)\n"

        for y in range(H):
            res = "["
            for x in range(W):
                if torch.allclose(reference[c, y, x], result[c, y, x], rtol, atol):
                    if not booldata:
                        res += " .    "
                    else:
                        res += " . "
                else:
                    if not booldata:
                        diff = (reference[c, y, x] - result[c, y, x]) / 10 ** (int(exp))
                        res += f"{diff: .2f} "
                    else:
                        diff = reference[c, y, x] ^ result[c, y, x]
                        res += " X "

                # Break early for large widths
                if x == 16:
                    res += "..."
                    break

            res += "]\n"
            output_str += res

    return output_str


def _print_elements(
    result: torch.Tensor,
    reference: torch.Tensor,
    C: int,
    H: int,
    W: int,
    rtol: float,
    atol: float,
) -> str:
    output_str = ""
    for y in range(H):
        res = "["
        for x in range(W):
            result_channels = result[:, y, x]
            reference_channels = reference[:, y, x]

            n_errors = 0
            for a, b in zip(result_channels, reference_channels):
                if not torch.allclose(a, b, rtol, atol):
                    n_errors = n_errors + 1

            if n_errors == 0:
                res += ". "
            else:
                res += f"{n_errors} "

            # Break early for large widths
            if x == 16:
                res += "..."
                break

        res += "]\n"
        output_str += res

    return output_str


def print_error_diffs(
    tester_or_result: Any,
    result_or_reference: TensorLike,
    reference: TensorLike | None = None,
    # Force remaining args to be keyword-only to keep the two positional call patterns unambiguous.
    *,
    quantization_scale: float | None = None,
    atol: float = 1e-03,
    rtol: float = 1e-03,
    qtol: float = 0,
) -> None:
    """
    Prints the error difference between a result tensor and a reference tensor in NCHW format.
    Certain formatting rules are applied to clarify errors:

    - Batches are only expanded if they contain errors.
        -> Shows if errors are related to batch handling
    - If errors appear in all channels, only the number of errors in each HW element are printed.
        -> Shows if errors are related to HW handling
    - If at least one channel is free from errors, or if C==1, errors are printed channel by channel
        -> Shows if errors are related to channel handling or single errors such as rounding/quantization errors

    Example output of shape (3,3,2,2):

        ############################ ERROR DIFFERENCE #############################
        BATCH 0
        .
        BATCH 1
        [. . ]
        [. 3 ]
        BATCH 2
        channel 1 (e-03)
        [ 1.85  .    ]
        [ .     9.32 ]

        MEAN      MEDIAN    MAX       MIN    (error as % of reference output range)
        60.02%    55.73%    100.17%   19.91%
        ###########################################################################


    """
    if reference is None:
        result = _ensure_tensor(cast(TensorLike, tester_or_result))
        reference_tensor = _ensure_tensor(result_or_reference)
    else:
        result = _ensure_tensor(result_or_reference)
        reference_tensor = _ensure_tensor(reference)

    if result.shape != reference_tensor.shape:
        raise ValueError(
            f"Output needs to be of same shape: {result.shape} != {reference_tensor.shape}"
        )
    shape = result.shape

    match len(shape):
        case 4:
            N, C, H, W = (shape[0], shape[1], shape[2], shape[3])
        case 3:
            N, C, H, W = (1, shape[0], shape[1], shape[2])
        case 2:
            N, C, H, W = (1, 1, shape[0], shape[1])
        case 1:
            N, C, H, W = (1, 1, 1, shape[0])
        case 0:
            N, C, H, W = (1, 1, 1, 1)
        case _:
            raise ValueError("Invalid tensor rank")

    if quantization_scale is not None:
        atol += quantization_scale * qtol

    # Reshape tensors to 4D NCHW format
    result = torch.reshape(result, (N, C, H, W))
    reference_tensor = torch.reshape(reference_tensor, (N, C, H, W))

    output_str = ""
    for n in range(N):
        output_str += f"BATCH {n}\n"
        result_batch = result[n, :, :, :]
        reference_batch = reference_tensor[n, :, :, :]

        is_close = torch.allclose(result_batch, reference_batch, rtol, atol)
        if is_close:
            output_str += ".\n"
        else:
            channels_close: list[bool] = [False] * C
            for c in range(C):
                result_hw = result[n, c, :, :]
                reference_hw = reference_tensor[n, c, :, :]

                channels_close[c] = torch.allclose(result_hw, reference_hw, rtol, atol)

            if any(channels_close) or len(channels_close) == 1:
                output_str += _print_channels(
                    result[n, :, :, :],
                    reference_tensor[n, :, :, :],
                    channels_close,
                    C,
                    H,
                    W,
                    rtol,
                    atol,
                )
            else:
                output_str += _print_elements(
                    result[n, :, :, :],
                    reference_tensor[n, :, :, :],
                    C,
                    H,
                    W,
                    rtol,
                    atol,
                )
        if reference_batch.dtype == torch.bool or result_batch.dtype == torch.bool:
            mismatches = (reference_batch != result_batch).sum().item()
            total = reference_batch.numel()
            output_str += f"(BOOLEAN tensor) {mismatches} / {total} elements differ ({mismatches / total:.2%})\n"

    # Only compute numeric error metrics if tensor is not boolean
    if reference_tensor.dtype != torch.bool and result.dtype != torch.bool:
        reference_range = torch.max(reference_tensor) - torch.min(reference_tensor)
        diff = torch.abs(reference_tensor - result).flatten()
        diff = diff[diff.nonzero()]
        if not len(diff) == 0:
            diff_percent = diff / reference_range
            output_str += "\nMEAN      MEDIAN    MAX       MIN    (error as % of reference output range)\n"
            output_str += f"{torch.mean(diff_percent):<8.2%}  {torch.median(diff_percent):<8.2%}  {torch.max(diff_percent):<8.2%}  {torch.min(diff_percent):<8.2%}\n"

    # Over-engineer separators to match output width
    lines = output_str.split("\n")
    line_length = [len(line) for line in lines]
    longest_line = max(line_length)
    title = "# ERROR DIFFERENCE #"
    separator_length = max(longest_line, len(title))

    pre_title_length = max(0, ((separator_length - len(title)) // 2))
    post_title_length = max(0, ((separator_length - len(title) + 1) // 2))
    start_separator = (
        "\n" + "#" * pre_title_length + title + "#" * post_title_length + "\n"
    )
    output_str = start_separator + output_str
    end_separator = "#" * separator_length + "\n"
    output_str += end_separator

    logger.error(output_str)


def dump_error_output(
    tester: Any,
    reference_output: TensorLike,
    stage_output: TensorLike,
    quantization_scale: float | None = None,
    atol: float = 1e-03,
    rtol: float = 1e-03,
    qtol: float = 0,
) -> None:
    """
    Prints Quantization info and error tolerances, and saves the differing tensors to disc.
    """
    # Capture assertion error and print more info
    banner = "=" * 40 + "TOSA debug info" + "=" * 40
    logger.error(banner)
    path_to_tosa_files = tester.compile_spec.get_intermediate_path()

    if path_to_tosa_files is None:
        path_to_tosa_files = tempfile.mkdtemp(prefix="executorch_result_dump_")

    export_stage = tester.stages.get(StageType.EXPORT, None)
    quantize_stage = tester.stages.get(StageType.QUANTIZE, None)
    if export_stage is not None and quantize_stage is not None:
        output_node = export_stage.artifact.graph_module.graph.output_node()
        qp_input = get_input_quantization_params(export_stage.artifact)
        qp_output = get_output_quantization_params(output_node)
        logger.error(f"Input QuantArgs: {qp_input}")
        logger.error(f"Output QuantArgs: {qp_output}")

    logger.error(f"{path_to_tosa_files=}")
    import os

    torch.save(
        stage_output,
        os.path.join(path_to_tosa_files, "torch_tosa_output.pt"),
    )
    torch.save(
        reference_output,
        os.path.join(path_to_tosa_files, "torch_ref_output.pt"),
    )
    logger.error(f"{atol=}, {rtol=}, {qtol=}")


if __name__ == "__main__":

    """This is expected to produce the example output of print_diff"""
    torch.manual_seed(0)
    a = torch.rand(3, 3, 2, 2) * 0.01
    b = a.clone().detach()
    logger.info(b)

    # Errors in all channels in element (1,1)
    a[1, :, 1, 1] = 0
    # Errors in (0,0) and (1,1) in channel 1
    a[2, 1, 1, 1] = 0
    a[2, 1, 0, 0] = 0

    print_error_diffs(a, b)
