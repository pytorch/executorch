# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import tempfile

import torch
from executorch.backends.arm.test.runner_utils import (
    _get_input_quantization_params,
    _get_output_node,
    _get_output_quantization_params,
)

from executorch.backends.xnnpack.test.tester.tester import Export, Quantize

logger = logging.getLogger(__name__)


def _print_channels(result, reference, channels_close, C, H, W, rtol, atol):

    output_str = ""
    for c in range(C):
        if channels_close[c]:
            continue

        max_diff = torch.max(torch.abs(reference - result))
        exp = f"{max_diff:2e}"[-3:]
        output_str += f"channel {c} (e{exp})\n"

        for y in range(H):
            res = "["
            for x in range(W):
                if torch.allclose(reference[c, y, x], result[c, y, x], rtol, atol):
                    res += " .    "
                else:
                    diff = (reference[c, y, x] - result[c, y, x]) / 10 ** (int(exp))
                    res += f"{diff: .2f} "

                # Break early for large widths
                if x == 16:
                    res += "..."
                    break

            res += "]\n"
            output_str += res

    return output_str


def _print_elements(result, reference, C, H, W, rtol, atol):
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
    tester,
    result: torch.Tensor | tuple,
    reference: torch.Tensor | tuple,
    quantization_scale=None,
    atol=1e-03,
    rtol=1e-03,
    qtol=0,
):
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

    if isinstance(reference, tuple):
        reference = reference[0]
    if isinstance(result, tuple):
        result = result[0]

    if not result.shape == reference.shape:
        raise ValueError("Output needs to be of same shape")
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
        case _:
            raise ValueError("Invalid tensor rank")

    if quantization_scale is not None:
        atol += quantization_scale * qtol

    # Reshape tensors to 4D NCHW format
    result = torch.reshape(result, (N, C, H, W))
    reference = torch.reshape(reference, (N, C, H, W))

    output_str = ""
    for n in range(N):
        output_str += f"BATCH {n}\n"
        result_batch = result[n, :, :, :]
        reference_batch = reference[n, :, :, :]
        is_close = torch.allclose(result_batch, reference_batch, rtol, atol)
        if is_close:
            output_str += ".\n"
        else:
            channels_close = [None] * C
            for c in range(C):
                result_hw = result[n, c, :, :]
                reference_hw = reference[n, c, :, :]

                channels_close[c] = torch.allclose(result_hw, reference_hw, rtol, atol)

            if any(channels_close) or len(channels_close) == 1:
                output_str += _print_channels(
                    result[n, :, :, :],
                    reference[n, :, :, :],
                    channels_close,
                    C,
                    H,
                    W,
                    rtol,
                    atol,
                )
            else:
                output_str += _print_elements(
                    result[n, :, :, :], reference[n, :, :, :], C, H, W, rtol, atol
                )

    reference_range = torch.max(reference) - torch.min(reference)
    diff = torch.abs(reference - result).flatten()
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
    tester,
    reference_output,
    stage_output,
    quantization_scale=None,
    atol=1e-03,
    rtol=1e-03,
    qtol=0,
):
    """
    Prints Quantization info and error tolerances, and saves the differing tensors to disc.
    """
    # Capture assertion error and print more info
    banner = "=" * 40 + "TOSA debug info" + "=" * 40
    logger.error(banner)
    path_to_tosa_files = tester.runner_util.intermediate_path

    if path_to_tosa_files is None:
        path_to_tosa_files = tempfile.mkdtemp(prefix="executorch_result_dump_")

    export_stage = tester.stages.get(tester.stage_name(Export), None)
    quantize_stage = tester.stages.get(tester.stage_name(Quantize), None)
    if export_stage is not None and quantize_stage is not None:
        output_node = _get_output_node(export_stage.artifact)
        qp_input = _get_input_quantization_params(export_stage.artifact)
        qp_output = _get_output_quantization_params(export_stage.artifact, output_node)
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
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    """ This is expected to produce the example output of print_diff"""
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
