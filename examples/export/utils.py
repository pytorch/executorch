# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import executorch.exir as exir


_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)

# Explicitly force the activation of the IR validator
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
)


def export_to_edge(
    model,
    method_name,
    example_inputs,
    capture_config=_CAPTURE_CONFIG,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
):
    m = model.eval()
    f = getattr(m, method_name)
    edge = exir.capture(f, example_inputs, capture_config).to_edge(edge_compile_config)
    logging.info(f"Exported graph:\n{edge.exported_program.graph}")
    return edge


def export_to_exec_prog(
    model,
    method_name,
    example_inputs,
    capture_config=_CAPTURE_CONFIG,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    backend_config=None,
):
    edge_m = export_to_edge(model, method_name, example_inputs, capture_config, edge_compile_config)
    exec_prog = edge_m.to_executorch(backend_config)
    return exec_prog


def save_pte_program(buffer, model_name):
    filename = f"{model_name}.pte"
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")
