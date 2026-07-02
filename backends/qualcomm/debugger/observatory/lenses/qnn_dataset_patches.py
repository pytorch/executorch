# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QNN dataset patches for AccuracyLens.

Installs monkey-patches on executorch.examples.qualcomm.utils dataset functions
to capture targets and task type for accuracy evaluation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.devtools.observatory.lenses.accuracy import AccuracyLens


def install_qnn_dataset_patches(cls: type[AccuracyLens]) -> None:
    """Install QNN dataset capture patches on AccuracyLens."""
    try:
        import executorch.examples.qualcomm.utils as utils_module

        if hasattr(utils_module, "get_imagenet_dataset"):
            original = utils_module.get_imagenet_dataset
            cls._originals["get_imagenet_dataset"] = original

            def patched_imagenet(*args, **kwargs):
                inputs, targets = original(*args, **kwargs)
                cls._captured_targets = targets
                cls._task_type = "classification"
                logging.info(
                    "[AccuracyLens] Captured ImageNet targets (%d samples)",
                    len(targets),
                )
                return inputs, targets

            utils_module.get_imagenet_dataset = patched_imagenet
            logging.info("[AccuracyLens] Installed patch: get_imagenet_dataset")

        if hasattr(utils_module, "get_masked_language_model_dataset"):
            original_mlm = utils_module.get_masked_language_model_dataset
            cls._originals["get_masked_language_model_dataset"] = original_mlm

            def patched_mlm(*args, **kwargs):
                inputs, targets = original_mlm(*args, **kwargs)
                cls._captured_targets = targets
                cls._task_type = "mlm"
                logging.info(
                    "[AccuracyLens] Captured MLM targets (%d samples)",
                    len(targets),
                )
                return inputs, targets

            utils_module.get_masked_language_model_dataset = patched_mlm
            logging.info(
                "[AccuracyLens] Installed patch: get_masked_language_model_dataset"
            )

        def _uninstall():
            try:
                for key, orig in cls._originals.items():
                    if hasattr(utils_module, key):
                        setattr(utils_module, key, orig)
            except Exception:
                pass

        cls._dataset_uninstallers.append(_uninstall)
    except ImportError:
        logging.debug(
            "[AccuracyLens] qualcomm utils not available, skipping dataset patches"
        )
