# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script runs the provided command from a symlinked version of the active
# working directory, in order to minimize path lengths and work around long
# path limitations on Windows.

set -x

PWSH_SCRIPT = "& {
    \$symlinkDir = Join-Path -Path \$env:GITHUB_WORKSPACE -ChildPath \"et-build\"
    New-Item -ItemType SymbolicLink -Path \$symlinkDir -Target $PWD
    Write-Host \$symlinkDir
}"

SYMLINK_DIR=`powershell -Command "$PWSH_SCRIPT"`
cd $SYMLINK_DIR
$1 ${@:2}
