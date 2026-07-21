# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export MobileNetV2 through the QNN backend and render its ETRecord as a compare HTML.

Requires a working QNN SDK and Python >= 3.11 with ``fast-sugiyama``:

    export QNN_SDK_ROOT=/path/to/qairt
    python -m executorch.devtools.fx_viewer.examples.demo_etrecord_compare_qnn

Because the QNN pipeline supplies ``transform_passes`` to
``to_edge_transform_and_lower``, the resulting bundle also contains an
``edge_after_transform`` stored program in ``graph_map`` — so the compare
view has three panes: aten, edge_after_transform, edge dialect (colored by
backend).
"""

from __future__ import annotations

import argparse
import os

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools.fx_viewer import export_etrecord_to_html
from executorch.examples.models.mobilenet_v2 import MV2Model


def _generate_etrecord(path: str, soc_model: QcomChipset) -> None:
    model = MV2Model().get_eager_model().eval()
    sample = MV2Model().get_example_inputs()
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=soc_model,
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
    edge_mgr = to_edge_transform_and_lower_to_qnn(
        model, sample, compile_spec, generate_etrecord=True
    )
    edge_mgr.to_executorch().get_etrecord().save(path)
    print(f"Saved ETRecord to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etrecord", default="mv2_qnn.etrecord")
    parser.add_argument("--html", default="mv2_etrecord_compare_qnn.html")
    parser.add_argument(
        "--soc",
        default="SM8650",
        help="Snapdragon chip enum name (see QcomChipset)",
    )
    parser.add_argument(
        "--reuse-etrecord",
        action="store_true",
        help="skip regeneration if the ETRecord file already exists",
    )
    args = parser.parse_args()

    soc = QcomChipset[args.soc]
    if not (args.reuse_etrecord and os.path.exists(args.etrecord)):
        _generate_etrecord(args.etrecord, soc)

    export_etrecord_to_html(
        args.etrecord,
        args.html,
        title="MV2 ETRecord — QNN (aten → edge_after_transform → edge)",
    )


if __name__ == "__main__":
    main()
