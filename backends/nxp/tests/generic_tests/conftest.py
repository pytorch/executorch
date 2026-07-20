# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests comparing against the TFLite reference are marked `@requires_tflite`
# (a `skipif` on `tflite is None`). Where the tflite/tensorflow interpreter is
# not importable, deselect them rather than letting the `skipif` skip them:
# internal CI (TestX) flags a consistently-skipping test as broken/disabled,
# while a deselected test produces no signal at all.

# Must stay in sync with the `reason=` of `requires_tflite` in test_quantizer.py.
_REQUIRES_TFLITE_REASON = "tensorflow/tflite not available"


def _is_tflite_gated_and_unavailable(item) -> bool:
    for marker in item.iter_markers(name="skipif"):
        condition = marker.args[0] if marker.args else False
        if condition and marker.kwargs.get("reason") == _REQUIRES_TFLITE_REASON:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    selected = []
    deselected = []
    for item in items:
        if _is_tflite_gated_and_unavailable(item):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
