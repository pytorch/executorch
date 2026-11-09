# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pytest
import torch

from examples.arm.mobilesam_prompt_segmentation_example_ethos_u.model_export.export_mobilesam import (
    iou,
    mask,
)
from examples.arm.mobilesam_prompt_segmentation_example_ethos_u.runtime.visualize_fvp_output import (
    load_mask,
    OUTPUT_SIZE,
)


def test_mask_and_iou() -> None:
    first = mask(torch.tensor([[[[-1.0, 1.0], [1.0, -1.0]]]]))
    second = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    assert first.tolist() == [[0, 1], [1, 0]]
    assert iou(first, second) == pytest.approx(1 / 3)


def test_load_mask(tmp_path: Path) -> None:
    output = np.ones((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    output[0, 0] = -1
    path = tmp_path / "output.bin"
    output.tofile(path)

    loaded = load_mask(path)

    assert loaded.shape == (OUTPUT_SIZE, OUTPUT_SIZE)
    assert loaded[0, 0] == 0
    assert loaded[1, 1] == 1


def test_load_mask_rejects_wrong_output_size(tmp_path: Path) -> None:
    path = tmp_path / "output.bin"
    np.ones(3, dtype=np.float32).tofile(path)

    with pytest.raises(ValueError, match="Expected"):
        load_mask(path)
