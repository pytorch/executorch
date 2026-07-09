#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

apply_patch_if_needed () {
  local patch="$1"
  if git apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "Already applied: $(basename "$patch")"
  else
    git am "$patch"
  fi
}

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Clone repos if not already cloned
if [[ ! -d meta-arm/.git ]]; then
  git clone https://git.yoctoproject.org/git/meta-arm -b CORSTONE1000-2025.12 meta-arm
fi
if [[ ! -d systemready-patch/.git ]]; then
  git clone https://git.gitlab.arm.com/arm-reference-solutions/systemready-patch.git \
    -b topics/hugkam01/a320_with_ssh systemready-patch
fi

cd meta-arm/

git am --abort 2>/dev/null || true

# Apply patches if not already applied
# TODO: Remove these patches and the apply_patch_if_needed func once we bump the version
apply_patch_if_needed ../systemready-patch/embedded-a/corstone1000/remove_second_bank/0001-Platform-Corstone1000-Drop-secondary-bank-reservatio.patch
apply_patch_if_needed ../systemready-patch/embedded-a/corstone1000/add-ssh-to-a320/0001-kas-Build-Corstone-1000-a320-with-SSH.patch

cd ../

cp systemready-patch/embedded-a/corstone1000/ethos-u85_test/ethos-u85_test.yml meta-arm/kas/

mkdir -p meta-arm/kas/local
cat > meta-arm/kas/local/enable-ssh.yml <<'EOF'
header:
  version: 14

local_conf_header:
  enable-ssh: |
    CORE_IMAGE_EXTRA_INSTALL:append = " packagegroup-core-ssh-dropbear ssh-pregen-hostkeys"
    IMAGE_CLASSES:append = " dropbear-rcs-symlink"
    CORE_IMAGE_EXTRA_INSTALL:append:firmware = " packagegroup-core-ssh-dropbear ssh-pregen-hostkeys"
    IMAGE_CLASSES:append:firmware = " dropbear-rcs-symlink"
  zzz-ssh-autostart: |
    # Re-add host key generation after other config removes it.
    CORE_IMAGE_EXTRA_INSTALL:append = " ssh-pregen-hostkeys"
EOF

mkdir -p meta-arm/meta-arm-bsp/classes/
cat > meta-arm/meta-arm-bsp/classes/dropbear-rcs-symlink.bbclass <<'EOF'
ROOTFS_POSTPROCESS_COMMAND:append = " dropbear_rcs_symlink; "

dropbear_rcs_symlink() {
    if [ -e ${IMAGE_ROOTFS}/etc/init.d/dropbear ]; then
        rcs_dir=${IMAGE_ROOTFS}/etc/rcS.d
        install -d ${rcs_dir}

        rm -f ${rcs_dir}/S50dropbear ${rcs_dir}/S50dropbear.sh

        printf '%s\n' \
            '#!/bin/sh' \
            '' \
            'case "$1" in' \
            'start|"")' \
            '    if ! pidof dropbear >/dev/null 2>&1; then' \
            '        if [ -x /etc/init.d/dropbear ]; then' \
            '            /etc/init.d/dropbear start >/dev/console 2>&1 || \' \
            '                echo "Dropbear auto-start failed" >/dev/console' \
            '        fi' \
            '    fi' \
            '    ;;' \
            'esac' \
            > ${rcs_dir}/S50dropbear.sh

        chmod 0755 ${rcs_dir}/S50dropbear.sh
    fi
}
EOF

if [[ "${ARM_FVP_EULA_ACCEPT:-}" != "1" ]]; then
    echo "You must export ARM_FVP_EULA_ACCEPT=1 to accept the Arm FVP EULA."
    exit 1
fi

kas build meta-arm/kas/corstone1000-a320.yml:meta-arm/ci/debug.yml:meta-arm/kas/ethos-u85_test.yml:meta-arm/kas/local/enable-ssh.yml

kas shell "meta-arm/kas/corstone1000-fvp.yml:meta-arm/ci/debug.yml:meta-arm/kas/corstone1000-a320.yml:meta-arm/kas/local/enable-ssh.yml" -c "bitbake corstone1000-recovery-image -c populate_sdk"
