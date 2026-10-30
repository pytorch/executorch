# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Push, train, and run inference for the QNN/HTP joint-graph ToyModel.

Extends SimpleADB to push the custom trainer/inference binaries and model
artifacts, then run training followed by on-device inference.

Usage:
    python run_toy_model.py --device <serial> --soc_model SM8850 \\
        --build_folder /path/to/build-android

    # push only
    python run_toy_model.py ... --push_only

    # skip push, train only (re-run without re-pushing)
    python run_toy_model.py ... --skip_push --skip_inference

    # skip push, infer only (after training already done on device)
    python run_toy_model.py ... --skip_push --skip_train

All config.json fields are read from --config (default: toy_model/config.json).
QNN_SDK_ROOT must be set (same requirement as all other Qualcomm examples).
"""

import argparse
import logging
import os

from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)

if __package__:
    from .export.config import load_config
else:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "export"))
    from config import load_config

logger = logging.getLogger(__name__)

_DEVICE_ROOT_TMPL = "/data/local/tmp/{user}/qnn_toy_model_trainer"
_BIN_SUBDIR = "bin"
_ARTIFACTS_SUBDIR = "artifacts"


class TrainADB(SimpleADB):
    """SimpleADB subclass for the ToyModel on-device training example.

    chmod_binaries() and push_artifacts() finish what SimpleADB.push() leaves
    to do (chmod the pushed trainer/inference binaries; push the artifacts/
    files alongside them); train() / infer() then run the binaries from the
    bin/ subdir, referencing artifacts/ for config and manifest.
    """

    def __init__(self, qnn_config: QnnConfig, workspace: str):
        super().__init__(
            qnn_config=qnn_config,
            pte_path=[],  # ptes live in artifacts/, pushed separately below.
            workspace=os.path.join(workspace, _BIN_SUBDIR),
            runner="examples/qualcomm/training/toy_model/on-device/qnn_toy_model_trainer",
        )
        self.workspace_root = workspace

    def chmod_binaries(self):
        """chmod+x the trainer/inference binaries already pushed by push()."""
        if self.skip_push:
            return
        bin_dir = f"{self.workspace_root}/{_BIN_SUBDIR}"
        for binary in ("qnn_toy_model_trainer", "qnn_toy_model_inference"):
            self._adb(["shell", f"chmod +x {bin_dir}/{binary}"])

    def push_artifacts(self, artifact_files, pte_files):
        """Push config/manifest/weights/pte files to the artifacts/ subdir."""
        if self.skip_push:
            return
        art_dir = f"{self.workspace_root}/{_ARTIFACTS_SUBDIR}"
        self._adb(["shell", f"rm -rf {art_dir}"])
        self._adb(["shell", f"mkdir -p {art_dir}"])
        for f in artifact_files + pte_files:
            self._adb(["push", f, f"{art_dir}/"])

    def train(self):
        """Run qnn_toy_model_trainer on-device."""
        bin_dir = f"{self.workspace_root}/{_BIN_SUBDIR}"
        art_dir = f"{self.workspace_root}/{_ARTIFACTS_SUBDIR}"
        self.execute(
            custom_runner_cmd=" ".join(
                [
                    f"cd {bin_dir} &&",
                    "export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. &&",
                    f"./qnn_toy_model_trainer {art_dir}/config.json {art_dir}/manifest.json",
                ]
            )
        )

    def infer(self):
        """Run qnn_toy_model_inference on-device."""
        bin_dir = f"{self.workspace_root}/{_BIN_SUBDIR}"
        art_dir = f"{self.workspace_root}/{_ARTIFACTS_SUBDIR}"
        self.execute(
            custom_runner_cmd=" ".join(
                [
                    f"cd {bin_dir} &&",
                    "export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. &&",
                    f"./qnn_toy_model_inference {art_dir}/config.json {art_dir}/manifest.json",
                ]
            )
        )


def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(
        args.config if args.config else os.path.join(script_dir, "config.json")
    )

    weight_sharing = (
        args.weight_sharing
        if args.weight_sharing is not None
        else cfg.get("weight_sharing", False)
    )
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/artifacts")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    artifact_subdir = os.path.join(output_dir, "ws" if weight_sharing else "no_ws")

    assert os.path.isdir(
        artifact_subdir
    ), f"Artifact dir not found: {artifact_subdir}; run export_toy_model.py first."

    pte_files = [
        os.path.join(artifact_subdir, f)
        for f in os.listdir(artifact_subdir)
        if f.endswith(".pte")
    ]
    assert (
        pte_files
    ), f"No .pte found in {artifact_subdir}; run export_toy_model.py first."

    device_root = _DEVICE_ROOT_TMPL.format(user=os.environ.get("USER", "user"))

    qnn_config = QnnConfig.load_config(args)
    adb = TrainADB(qnn_config=qnn_config, workspace=device_root)

    inference_bin = os.path.join(
        args.build_folder,
        "examples/qualcomm/training/toy_model/on-device/qnn_toy_model_inference",
    )
    # Binaries + QNN libs go to bin/ (handled by SimpleADB.push via workspace=bin/).
    # Artifacts (ptes already in pte_path, plus config/manifest/weights) go to artifacts/.
    artifact_files = [
        os.path.join(script_dir, "config.json"),
        os.path.join(artifact_subdir, "manifest.json"),
        os.path.join(artifact_subdir, "init_weights.safetensors"),
    ]
    adb.push(files=[inference_bin])
    adb.chmod_binaries()
    adb.push_artifacts(artifact_files, pte_files)

    if not args.push_only:
        if not args.skip_train:
            adb.train()
        if not args.skip_inference:
            adb.infer()


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: config.json next to this script).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base artifacts directory (default: from config.json).",
    )
    parser.add_argument(
        "--weight_sharing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use weight-sharing variant (ws/ subdir). Default: from config.json.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="Skip on-device training (run inference only).",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        default=False,
        help="Skip on-device inference (train only).",
    )
    parser.add_argument(
        "--push_only",
        action="store_true",
        default=False,
        help="Push binaries/artifacts to device, but skip training and inference.",
    )
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
