#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Parse CLI options and configure logging before importing the exporter."""

import argparse
import logging

from executorch.examples.models import MODEL_NAME_TO_MODEL

logger = logging.getLogger(__name__)

TARGETS = [
    "ethos-u55-32",
    "ethos-u55-64",
    "ethos-u55-128",
    "ethos-u55-256",
    "ethos-u65-256",
    "ethos-u65-512",
    "ethos-u85-128",
    "ethos-u85-256",
    "ethos-u85-512",
    "ethos-u85-1024",
    "ethos-u85-2048",
    "vgf",
    "TOSA-1.0+INT",
    "TOSA-1.0+FP",
    "TOSA-1.0+INT+int16",
    "cortex-m0",
    "cortex-m0plus",
    "cortex-m3",
    "cortex-m4",
    "cortex-m7",
    "cortex-m23",
    "cortex-m33",
    "cortex-m35p",
    "cortex-m55",
    "cortex-m85",
]


def _configure_logging(debug: bool):
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
            force=True,
        )
        logging.getLogger("executorch.backends.arm").setLevel(logging.DEBUG)
        logging.getLogger("executorch.backends.cortex_m").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.WARNING, format="%(levelname)s :%(message)s", force=True
        )
        logging.getLogger("executorch.backends.arm").setLevel(logging.INFO)
        logging.getLogger("executorch.backends.cortex_m").setLevel(logging.INFO)
        logger.info(
            "Tip - For more verbose logs, use --debug. Use -i <path> to save artifacts to a given directory."
        )

    logging.getLogger("executorch.exir.backend.backend_api").setLevel(logging.INFO)
    logging.getLogger("executorch.backends.cortex_m.quantizer_reporter").setLevel(
        logging.DEBUG if debug else logging.INFO
    )

    logging.getLogger(
        "executorch.backends.arm.scripts.aot_arm_compiler_backend"
    ).setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Model file .py/.pth/.pt or a model from examples/models. Valid names: {set(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "--model_input",
        required=False,
        default=None,
        help="Provide model input .pt file, or python variable name",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing ArmBackend delegated model",
    )
    parser.add_argument(
        "--bundleio",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing BundleIO bpte file with input/output test/ref data.",
    )
    parser.add_argument(
        "--etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing a etrecord file.",
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        required=False,
        default="ethos-u55-128",
        choices=TARGETS,
        help=f"Target backend. For delegated models: Ethos-U/VGF/TOSA variants. For non-delegated: cortex-m<variant> (CMSIS-NN portable kernels). Valid targets: {TARGETS}",
    )
    parser.add_argument(
        "--cortex-m-explicit-layout",
        action="store_true",
        help=(
            "Use explicit NCHW/NHWC permutes for Cortex-M instead of dim-order "
            "operators. This is an experimental Cortex-M-only option."
        ),
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce a quantized model",
    )
    parser.add_argument(
        "--calibration_data",
        required=False,
        default=None,
        help=(
            "Optional calibration data file or directory. If a directory is "
            "provided, up to 1000 samples are used for calibration. "
            "Supported files: .pt/.pth. If not provided,"
            "quantized models are calibrated on their example inputs."
        ),
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        default=None,
        help="Provide path to custom .so library.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set the logging level to debug."
    )
    parser.add_argument(
        "-i",
        "--intermediates",
        action="store",
        required=False,
        help="Store intermediate output (like TOSA artifacts) somewhere.",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=False,
        help="Filename (if .pte or .bpte is used) or a folder for outputs, if not specified the default is to place files in cwd.",
    )
    parser.add_argument(
        "--system_config",
        required=False,
        default=None,
        help="System configuration to select from the Vela configuration file (see vela.ini). This option must match the selected target, default is for an optimal system 'Ethos_U55_High_End_Embedded'/ 'Ethos_U65_High_End' / 'Ethos_U85_SYS_DRAM_Mid'",
    )
    parser.add_argument(
        "--memory_mode",
        required=False,
        default=None,
        help="Memory mode to select from the Vela configuration file (see vela.ini). Default is 'Shared_Sram' for Ethos-U55 targets and 'Sram_Only' for Ethos-U65 and Ethos-U85 targets",
    )
    parser.add_argument(
        "--plan-delegate-scratch",
        action="store_true",
        help="Declare Ethos-U delegate scratch for ExecuTorch memory planning. "
        "Use with the bare-metal/RTOS core-driver runtime; Linux Direct Drive "
        "allocates its own scratch and retains the unused planned reservation.",
    )
    parser.add_argument(
        "--max_scratch_size",
        type=int,
        default=None,
        help="Maximum Ethos-U delegate scratch size in bytes. Defaults to 2097152 "
        "for U55 Shared_Sram with Ethos_U55_High_End_Embedded and Arm/vela.ini "
        "(the Corstone-300 test configuration); unset for other configurations. "
        "Override only to match the deployment platform's scratch capacity.",
    )
    parser.add_argument(
        "--config",
        required=False,
        default="Arm/vela.ini",
        help="Specify custom vela configuration file (vela.ini) for Ethos-U targets.",
    )
    parser.add_argument(
        "--non_strict_export",
        dest="strict_export",
        required=False,
        action="store_false",
        help="Disable strict checking while exporting models.",
    )
    parser.add_argument(
        "--enable_debug_mode",
        required=False,
        choices=["json", "tosa"],
        help="Flag to enable ATen-to-TOSA debug mode and dumping of Vela's debug database.",
    )
    parser.add_argument(
        "--direct_drive",
        action="store_true",
        required=False,
        default=False,
        help="Flag for enabling direct drive.",
    )
    args = parser.parse_args()

    if args.plan_delegate_scratch and "ethos-u" not in args.target:
        parser.error("--plan-delegate-scratch requires an Ethos-U target")

    if args.calibration_data is not None and not args.quantize:
        parser.error("--calibration_data requires --quantize to be enabled.")

    return args


def main() -> None:
    """Parse arguments, configure logging, then import and run the exporter."""
    args = _get_args()
    _configure_logging(args.debug)

    from executorch.backends.arm.scripts.aot_arm_compiler_backend import export

    export(args)


if __name__ == "__main__":
    main()
