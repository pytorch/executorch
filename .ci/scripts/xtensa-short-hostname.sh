#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Re-exec the calling script under a short hostname when the real one is long
# enough to break the toolchain's licence checkout.
#
# RJ-2025.5 xt-clang builds its licence feature name (XT_XCC_TIE_<config id>)
# through a buffer that the hostname overruns at 47 characters. The computed
# name then does not match the one the licence grants, and FlexNet answers "No
# such feature exists", saying nothing about the host. RI-2022.9 and xt-run are
# unaffected, so this only guards the RJ toolchains. A CI pod's hostname is the
# runner label plus 28 characters, which puts any label over 18 past the limit.
#
# Source this from a script that invokes the toolchain; do not execute it.

XTENSA_MAX_HOSTNAME="${XTENSA_MAX_HOSTNAME:-46}"
XTENSA_SHORT_HOSTNAME=xtensa-ci

if [[ "${TOOLCHAIN_VER:-}" == RJ-* && -z "${XTENSA_HOSTNAME_SHORTENED:-}" ]]; then
  _host_name=$(hostname)
  if [[ ${#_host_name} -gt ${XTENSA_MAX_HOSTNAME} ]]; then
    export XTENSA_HOSTNAME_SHORTENED=1
    # -U -r maps our uid to root inside a new user namespace, which is what
    # lets an unprivileged container process own a UTS namespace and rename it.
    if unshare -Uur true 2>/dev/null; then
      echo "Hostname is ${#_host_name} chars, over the ${XTENSA_MAX_HOSTNAME} that ${TOOLCHAIN_VER} tolerates;" \
           "re-running as '${XTENSA_SHORT_HOSTNAME}'."
      exec unshare -Uur bash -c \
        "hostname ${XTENSA_SHORT_HOSTNAME}; exec \"\$0\" \"\$@\"" "$0" "$@"
    fi
    echo "ERROR: hostname '${_host_name}' is ${#_host_name} chars; ${TOOLCHAIN_VER} needs ${XTENSA_MAX_HOSTNAME} or fewer," >&2
    echo "       and unshare(2) is not permitted here to shorten it. Use a host with a shorter name." >&2
    exit 1
  fi
fi
