# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Generate PTE files for all convolution layer configurations from conv_layer_configs.h.

Each layer's config_key encodes: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil

For each unique configuration, this script:
  1. Creates a QuantizedConv model with the matching parameters
  2. Exports it via the Cadence AOT pipeline
  3. Copies the .pte from /tmp/ to operator_and_model_testing/conv2d/pte/
  4. Stores logs to operator_and_model_testing/conv2d/output/
"""

import logging
import os
import sys
import shutil
import tempfile
import traceback
from typing import cast, Sequence

import torch

from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from executorch.backends.cadence.aot.export_example import export_model

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# ============================================================================
# Selected layer configurations for specific ResNet layers
# config_key format: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil
# ============================================================================
LAYER_CONFIGS = [
    # layer_id, layer_name, config_key
    (0,  "conv1",         "3_64_64_64_7_7_32_32_2_2_3_1"),
    (1,  "conv2.1",       "64_16_16_64_3_3_16_16_1_1_1_1"),
    (2,  "conv4b.1",      "64_16_16_128_3_3_8_8_2_2_1_1"),
    (3,  "conv4b.2",      "128_8_8_128_3_3_8_8_1_1_1_1"),
    (4,  "conv4a.1",      "64_16_16_128_1_1_8_8_2_2_0_1"),
    (5,  "conv6b.1",      "128_8_8_256_3_3_4_4_2_2_1_1"),
    (6,  "conv6b.2",      "256_4_4_256_3_3_4_4_1_1_1_1"),
    (7,  "conv6a.1",      "128_8_8_256_1_1_4_4_2_2_0_1"),
    (8,  "conv8b.1",      "256_4_4_512_3_3_2_2_2_2_1_1"),
    (9,  "conv8b.2",      "512_2_2_512_3_3_2_2_1_1_1_1"),
    (10, "conv8a.1",      "256_4_4_512_1_1_2_2_2_2_0_1"),
]


def parse_config_key(config_key: str):
    """Parse config_key: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil"""
    parts = config_key.split("_")
    assert len(parts) == 12, f"Expected 12 fields in config_key, got {len(parts)}: {config_key}"
    ic, ih, iw, oc, kh, kw, oh, ow, sy, sx, pad, dil = [int(p) for p in parts]
    return ic, ih, iw, oc, kh, kw, oh, ow, sy, sx, pad, dil


def generate_pte_for_layer(layer_id, layer_name, config_key, pte_dir, output_dir):
    """Generate a PTE for a single conv layer configuration."""
    ic, ih, iw, oc, kh, kw, oh, ow, sy, sx, pad, dil = parse_config_key(config_key)

    # Sanitize name for filenames
    safe_name = layer_name.replace(".", "_")
    file_name = f"conv2d_layer{layer_id}_{safe_name}"
    log_file = os.path.join(output_dir, f"{file_name}.log")

    logging.info(f"=== Layer {layer_id}: {layer_name} ({config_key}) ===")
    logging.info(f"  in_channels={ic}, out_channels={oc}, kernel=({kh},{kw}), "
                 f"stride=({sy},{sx}), padding=({pad},{pad}), dilation=({dil},{dil})")
    logging.info(f"  input_shape=(1,{ic},{ih},{iw}), output_shape=(1,{oc},{oh},{ow})")

    # Redirect logging to file
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        class QuantizedConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(
                    ic, oc, (kh, kw),
                    stride=(sy, sx),
                    padding=(pad, pad),
                    dilation=(dil, dil),
                    groups=1,
                    bias=True,
                )

            def forward(self, x: torch.Tensor):
                return self.conv2d(x)

        model = QuantizedConv()
        model.eval()

        input_shape = (1, ic, ih, iw)
        example_inputs = (torch.randn(cast(Sequence[int], input_shape)),)

        # Create a dedicated temp working dir for this layer
        working_dir = tempfile.mkdtemp(dir="/tmp", prefix=f"conv2d_{safe_name}_")

        # Export the model — saves <file_name>.pte and .bpte to working_dir
        exec_prog = export_model(model, example_inputs, file_name, working_dir=working_dir)

        # Copy PTE from working_dir to the pte output directory
        src_pte = os.path.join(working_dir, f"{file_name}.pte")
        if os.path.exists(src_pte):
            dst_pte = os.path.join(pte_dir, f"{file_name}.pte")
            shutil.copy2(src_pte, dst_pte)
            logging.info(f"  Copied PTE: {src_pte} -> {dst_pte}")
        else:
            logging.warning(f"  PTE file not found at {src_pte}")

        logging.info(f"  Layer {layer_id} ({layer_name}): SUCCESS")
        return True

    except Exception as e:
        logging.error(f"  Layer {layer_id} ({layer_name}): FAILED - {e}")
        logging.error(traceback.format_exc())
        return False

    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pte_dir = os.path.join(script_dir, "pte")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(pte_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Deduplicate by config_key — layers with the same config produce identical PTEs
    # but we still generate one per layer_id for traceability
    total = len(LAYER_CONFIGS)
    success = 0
    failed = 0
    failed_layers = []

    logging.info(f"Generating PTEs for {total} conv2d layer configurations")
    logging.info(f"PTE output dir: {pte_dir}")
    logging.info(f"Log output dir: {output_dir}")

    for layer_id, layer_name, config_key in LAYER_CONFIGS:
        ok = generate_pte_for_layer(layer_id, layer_name, config_key, pte_dir, output_dir)
        if ok:
            success += 1
        else:
            failed += 1
            failed_layers.append((layer_id, layer_name))

    # Summary
    logging.info("=" * 60)
    logging.info(f"SUMMARY: {success}/{total} succeeded, {failed}/{total} failed")
    if failed_layers:
        for lid, lname in failed_layers:
            logging.info(f"  FAILED: layer {lid} ({lname})")
    logging.info("=" * 60)

    # Save summary to output dir
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Failed: {failed}\n")
        for lid, lname in failed_layers:
            f.write(f"FAILED: layer {lid} ({lname})\n")

    # List generated PTEs
    pte_files = sorted(os.listdir(pte_dir))
    pte_files = [f for f in pte_files if f.endswith(".pte")]
    logging.info(f"Generated {len(pte_files)} PTE files in {pte_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
