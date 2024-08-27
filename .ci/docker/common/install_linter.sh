#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# NB: Install all linter dependencies, the caching of lintrunner init could be
# done after Executorch becomes public
pip_install -r requirements-lintrunner.txt

# Install google-java-format
curl -L --retry 3 https://github.com/google/google-java-format/releases/download/v1.23.0/google-java-format_linux-x86-64 > /opt/google-java-format
chmod +x /opt/google-java-format
