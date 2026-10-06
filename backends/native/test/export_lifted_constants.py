# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from executorch.backends.native.serialization import serialize_graph
from executorch.backends.native.test.utils import lifted_constant_program
from executorch.exir.native.native import NativeProgramManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ep = lifted_constant_program((torch.arange(4.0), torch.arange(4.0)))
    graph, constants = serialize_graph(
        ep.graph_module, ep.graph_signature, ep.state_dict, ep.constants
    )
    NativeProgramManager(graph, constants).save(args.output)
