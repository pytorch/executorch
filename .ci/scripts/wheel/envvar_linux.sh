# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/envvar_base.sh"

# The reusable smoke installer selects PyTorch's index; ET also needs PyPI packages.
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:+${PIP_EXTRA_INDEX_URL} }https://pypi.org/simple"
