#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch.utils.cpp_extension

sources = [
    "MPSGraphInterface.mm",
    "OperationUtils.mm",
    "Bindings.mm",
]

ops = [
    "ConvolutionOps.mm",
    "NormalizationOps.mm",
    "ActivationOps.mm",
    "ReduceOps.mm",
    "ConstantOps.mm",
    "UnaryOps.mm",
    "BinaryOps.mm",
    "ClampOps.mm",
    "ShapeOps.mm",
    "LinearAlgebraOps.mm",
    "BitwiseOps.mm",
    "PoolingOps.mm",
    "PadOps.mm",
    "RangeOps.mm",
    "Indexing.mm",
]

SOURCES_PATH = "backends/apple/mps/"
final_sources = [SOURCES_PATH + "utils/" + source for source in sources] + [
    SOURCES_PATH + "operations/" + op for op in ops
]

graph_bindings = torch.utils.cpp_extension.load(
    name="MPSGraphBindings",
    sources=final_sources,
    extra_include_paths=[SOURCES_PATH],
    verbose=False,
)
