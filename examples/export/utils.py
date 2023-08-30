# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import executorch.exir as exir

# Using dynamic shape does not allow us to run graph_module returned by
# to_executorch for mobilenet_v3.
# Reason is that there memory allocation ops with symbolic shape nodes.
# and when evaulating shape, it doesnt seem that we presenting them with shape env
# that contain those variables.
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def export_to_edge(model, example_inputs):
    m = model.eval()
    edge = exir.capture(m, example_inputs, _CAPTURE_CONFIG).to_edge(
        _EDGE_COMPILE_CONFIG
    )
    logging.info(f"Exported graph:\n{edge.exported_program.graph}")
    return edge


def export_to_pte(model_name, model, example_inputs):
    edge = export_to_edge(model, example_inputs)
    exec_prog = edge.to_executorch()
    return exec_prog.buffer


def save_pte_program(buffer, model_name):
    filename = f"{model_name}.pte"
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")
