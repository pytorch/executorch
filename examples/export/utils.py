# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.exir as exir

# Using dynamic shape does not allow us to run graph_module returned by
# to_executorch for mobilenet_v3.
# Reason is that there memory allocation ops with symbolic shape nodes.
# and when evaulating shape, it doesnt seem that we presenting them with shape env
# that contain those variables.
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)


def export_to_edge(model, example_inputs):
    m = model.eval()
    edge = exir.capture(m, example_inputs, _CAPTURE_CONFIG).to_edge()
    print("Exported graph:\n", edge.exported_program.graph)
    return edge
