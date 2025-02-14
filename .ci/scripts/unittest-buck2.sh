#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# TODO: expand this to //...
buck2 query //runtime/...

# TODO: expand the covered scope of Buck targets.
buck2 build //runtime/core/portable_type/...
buck2 test //runtime/core/portable_type/...
