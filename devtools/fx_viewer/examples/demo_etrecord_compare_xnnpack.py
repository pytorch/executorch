# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export MobileNetV2 through XNNPACK and render its ETRecord as a compare HTML.

Runnable in an ExecuTorch dev env (Python >= 3.11 with ``fast-sugiyama``):

    python -m executorch.devtools.fx_viewer.examples.demo_etrecord_compare_xnnpack

Outputs:
    mv2_etrecord_compare_xnnpack.html
"""

from __future__ import annotations

import argparse
import os

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.fx_viewer import export_etrecord_to_html
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.exir import to_edge_transform_and_lower


def _generate_etrecord(path: str) -> None:
    model = MV2Model().get_eager_model().eval()
    example_inputs = MV2Model().get_example_inputs()
    with torch.no_grad():
        ep = torch.export.export(model, example_inputs, strict=False)
    edge_mgr = to_edge_transform_and_lower(
        ep, partitioner=[XnnpackPartitioner()], generate_etrecord=True
    )
    edge_mgr.to_executorch().get_etrecord().save(path)
    print(f"Saved ETRecord to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etrecord", default="mv2_xnnpack.etrecord")
    parser.add_argument("--html", default="mv2_etrecord_compare_xnnpack.html")
    parser.add_argument(
        "--reuse-etrecord",
        action="store_true",
        help="skip regeneration if the ETRecord file already exists",
    )
    args = parser.parse_args()

    if not (args.reuse_etrecord and os.path.exists(args.etrecord)):
        _generate_etrecord(args.etrecord)

    export_etrecord_to_html(
        args.etrecord,
        args.html,
        title="MV2 ETRecord — XNNPACK (aten → edge)",
    )


if __name__ == "__main__":
    main()
