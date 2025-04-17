#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
last= rc=0
while IFS=: read -r f u; do
  if [ "$f" != "$last" ]; then
    [ -n "$last" ] && echo
    printf '%s:\n' "$f"
    last=$f
  fi
  if curl --fail -s -m10 -o /dev/null "$u"; then
    printf " ${green}[OK]${reset}  ${cyan}%s${reset}\n" "$u"
  else
    printf "${red}[FAIL]${reset} ${yellow}%s${reset}\n" "$u"
    rc=1
  fi
done < <(
  git --no-pager grep --no-color -I -o -E 'https?://[^[:space:]<>\")\{]+' \
    -- '*' \
    ':(exclude).*' ':(exclude)**/.*' ':(exclude)**/*.lock' ':(exclude)**/third-party/**' \
  | sed 's/[."\’]$//'
)
exit $rc
