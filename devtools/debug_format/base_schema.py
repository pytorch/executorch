# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Base Intermediate Representation for Developer Tools consumers
(e.g. TensorBoard, Terminal Debugger)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Base Representation of a generic node within a ModelGraph
@dataclass
class Node:
    name: str
    # Nodes that this Node consumes/in-edges
    inputs: Optional[List[Node]] = None
    # List of output shapes
    output_shapes: Optional[List[List[int]]] = None
    # Generic Node level metadata
    metadata: Optional[Dict[str, Any]] = None
    # Names of the arguments derived from the op schema:
    named_args: Optional[List[str]] = None


# Base Representation of an operator subgraph with metadata
@dataclass
class OperatorGraph:
    # Identifier used for grouping nodes (e.g. expand/minimize Module)
    graph_name: str
    # Nodes and Sub-Graphs
    elements: List[Node | OperatorGraph]
    # Graph Level Metadata
    metadata: Optional[Dict[str, Any]] = None


"""
Node SubClasses Types
"""


# Representation of a "Value" node within a ModelGraph
# i.e. Non-Operator Nodes
@dataclass
class ValueNode(Node):
    dtype: str = ""
    val: Optional[Any] = None


# Representation of an "OP" node within a ModelGraph
@dataclass
class OperatorNode(Node):
    op: Optional[str] = None
