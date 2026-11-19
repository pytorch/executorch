# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from executorch.backends.arm.arm_vela import (
    _persistent_init_image,
    _persistent_region_size,
)
from pytest import raises

_SHAPE = [1, 1, 1, 1, 1, 4]


def _savez_roundtrip(tmp_path, **arrays):
    # Go through np.savez/np.load as Vela's rawdata_writer does, which strips
    # trailing zero bytes from the per-variable data entries.
    path = tmp_path / "out_vela.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _variables(tmp_path, offsets, regions, data, elem_size=2):
    return _savez_roundtrip(
        tmp_path,
        variable_offset=offsets,
        variable_region=regions,
        variable_shape=[_SHAPE] * len(offsets),
        variable_elem_size=[elem_size] * len(offsets),
        variable_data=data,
        variable_data_size=[len(d) for d in data],
    )


def test_region_size_is_max_end_of_region_5_variables_aligned(tmp_path):
    data = _variables(tmp_path, [0, 32, 0], [5, 5, 1], [b"", b"", b""])
    assert _persistent_region_size(data) == 48


def test_no_variables_means_no_region():
    assert _persistent_region_size({}) == 0
    assert _persistent_init_image({}, 0) is None


def test_image_places_data_and_restores_trailing_zeros(tmp_path):
    # Read and write tensor of one variable at offset 0 (only the read tensor
    # has data, ending in zero bytes), and a second variable at 16.
    first = bytes([1, 2, 3, 4, 5, 0, 0, 0])
    second = bytes([0, 0, 9, 9, 9, 9, 9, 9])
    data = _variables(tmp_path, [0, 0, 16], [5, 5, 5], [first, b"", second])
    image = _persistent_init_image(data, 32)
    assert image is not None
    assert image[0:8] == first
    assert image[8:16] == bytes(8)
    assert image[16:24] == second
    assert image[24:32] == bytes(8)


def test_all_zero_image_is_left_out(tmp_path):
    data = _variables(tmp_path, [0, 0], [5, 5], [bytes(8), b""])
    assert _persistent_init_image(data, 16) is None


def test_initial_data_outside_persistent_region_is_rejected(tmp_path):
    data = _variables(tmp_path, [0], [1], [bytes([1] * 8)])
    with raises(ValueError, match="region 1"):
        _persistent_init_image(data, 16)


def test_initial_data_past_region_end_is_rejected(tmp_path):
    data = _variables(tmp_path, [12], [5], [bytes([1] * 8)])
    with raises(ValueError, match="exceeds"):
        _persistent_init_image(data, 16)
