# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional

import torch

from executorch.exir import ExportedProgram

from torch.fx.passes.utils.matcher_utils import InternalMatch

GetGraphFn = Callable[[], List[torch.fx.GraphModule]]
CreateReplacementFn = Callable[
    [ExportedProgram, torch.fx.GraphModule, InternalMatch], None
]


class PatternEntry:
    def __init__(
        self,
        get_graphs_fn: Optional[GetGraphFn] = None,
        create_replacement_fn: Optional[CreateReplacementFn] = None,
    ):
        self.get_graphs_fn = get_graphs_fn
        self.create_replacement_fn = create_replacement_fn

    def is_valid(self):
        return self.get_graphs_fn is not None and self.create_replacement_fn is not None


fusable_patterns: Dict[str, PatternEntry] = {}


def register_pattern_graph(pattern_name: str):
    def decorator(fn: GetGraphFn):
        if pattern_name not in fusable_patterns:
            fusable_patterns[pattern_name] = PatternEntry()

        fusable_patterns[pattern_name].get_graphs_fn = fn
        return fn

    return decorator


def register_pattern_replacement(pattern_name: str):
    def decorator(fn: CreateReplacementFn):
        if pattern_name not in fusable_patterns:
            fusable_patterns[pattern_name] = PatternEntry()

        fusable_patterns[pattern_name].create_replacement_fn = fn
        return fn

    return decorator
