# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import typing
from typing import Callable, Union

import numpy as np
import torch


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def distance(fn: Callable[[np.ndarray, np.ndarray], float]) -> Callable[
    [
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        typing.Union[np.ndarray, torch._tensor.Tensor],
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        typing.Union[np.ndarray, torch._tensor.Tensor],
    ],
    float,
]:
    # A distance decorator that performs all the necessary checkes before calculating
    # the distance between two N-D tensors given a function. This can be a RMS
    # function, maximum abs diff, or any kind of distance function.
    def wrapper(
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        a: Union[np.ndarray, torch.Tensor],
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        b: Union[np.ndarray, torch.Tensor],
    ) -> float:
        # convert a and b to np.ndarray type fp64
        a = to_np_arr_fp64(a)
        b = to_np_arr_fp64(b)

        # return NaN if shape mismatches
        if a.shape != b.shape:
            return np.nan

        # After we make sure shape matches, check if it's empty. If yes, return 0
        if a.size == 0:
            return 0

        # np.isinf and np.isnan returns a Boolean mask. Check if Inf or NaN occur at
        # the same places in a and b. If not, return NaN
        if np.any(np.isinf(a) != np.isinf(b)) or np.any(np.isnan(a) != np.isnan(b)):
            return np.nan

        # mask out all the values that are either Inf or NaN
        mask = np.isinf(a) | np.isnan(a)
        if np.any(mask):
            logging.warning("Found inf/nan in tensor when calculating the distance")

        a_masked = a[~mask]
        b_masked = b[~mask]

        # after masking, the resulting tensor might be empty. If yes, return 0
        if a_masked.size == 0:
            return 0

        # only compare the rest (those that are actually numbers) using the metric
        return fn(a_masked, b_masked)

    return wrapper


@distance
# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def rms(a: np.ndarray, b: np.ndarray) -> float:
    return ((a - b) ** 2).mean() ** 0.5


@distance
# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return np.abs(a - b).max()


@distance
# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def max_rel_diff(x: np.ndarray, x_ref: np.ndarray) -> float:
    return np.abs((x - x_ref) / x_ref).max()


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def to_np_arr_fp64(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        x = x.astype(np.float64)
    return x


# pyre-fixme[3]: Return type must be annotated.
def normalized_rms(
    # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    predicted: Union[np.ndarray, torch.Tensor],
    # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    ground_truth: Union[np.ndarray, torch.Tensor],
):
    num = rms(predicted, ground_truth)
    if num == 0:
        return 0
    den = np.linalg.norm(to_np_arr_fp64(ground_truth))
    return np.float64(num) / np.float64(den)
