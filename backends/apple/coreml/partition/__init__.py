# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

from .coreml_partitioner import CoreMLPartitioner, SingleOpCoreMLPartitioner

__all__ = [
    CoreMLPartitioner,
    SingleOpCoreMLPartitioner,
]
