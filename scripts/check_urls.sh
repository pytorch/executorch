#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
last_filepath=

while IFS=: read -r filepath url; do
  if [ "$filepath" != "$last_filepath" ]; then
    printf '\n%s:\n' "$filepath"
    last_filepath=$filepath
  fi
  code=$(curl -g -sSL -m30 \
    -A 'Mozilla/5.0' \
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
    --range 0-0 \
    -w '%{http_code}' -o /dev/null \
    "$url") || code=000
  if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then
    printf "${green}%s${reset} ${cyan}%s${reset}\n" "$code" "$url"
  else
    printf "${red}%s${reset} ${yellow}%s${reset}\n" "$code" "$url" >&2
    status=1
  fi
done < <(
  git --no-pager grep --no-color -I -P -o \
    '(?<!git\+)(?<!\$\{)https?://(?![^\s<>\")]*[\{\}\$])[^[:space:]<>\")\[\]\(]+' \
    -- '*' \
    ':(exclude).*' \
    ':(exclude)**/.*' \
    ':(exclude)**/*.lock' \
    ':(exclude)**/*.svg' \
    ':(exclude)**/*.xml' \
    ':(exclude)**/*.gradle*' \
    ':(exclude)**/*gradle*' \
    ':(exclude)**/third-party/**' \
  | sed -E 's/[^/[:alnum:]]+$//' \
  | grep -Ev '://(0\.0\.0\.0|127\.0\.0\.1|localhost)([:/])' \
  | grep -Ev 'fwdproxy:8080' \
  || true
)

exit $status
