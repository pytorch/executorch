# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

def flatten_directory_contents(path: str) -> Optional[bytes]: ...
def unflatten_directory_contents(bytes: bytes, path: str) -> bool: ...
