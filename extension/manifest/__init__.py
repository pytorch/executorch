# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.manifest._manifest import append_manifest, Manifest

__all__ = [
    "Manifest",
    "append_manifest",
]
