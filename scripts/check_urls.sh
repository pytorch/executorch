#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

curl --http1.1 -sS -o /dev/null -w '%{http_code}\n' \
  -H 'Accept-Language: en-US,en;q=0.9' \
  -H 'Accept-Encoding: gzip, deflate, br' \
  -H 'Referer: https://www.google.com/' \
  -H 'Connection: keep-alive' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36' \
  https://www.cadence.com/en_US/home.html

curl -c cookies.txt -b cookies.txt -sS -o /dev/null -w '%{http_code}\n' \
  -H 'Accept-Language: en-US,en;q=0.9' \
  -H 'Referer: https://www.google.com/' \
  -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36' \
  https://www.cadence.com/en_US/home.html

set -euo pipefail

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
last_filepath=

while IFS=: read -r filepath url; do
  if [ "$filepath" != "$last_filepath" ]; then
    printf '\n%s:\n' "$filepath"
    last_filepath=$filepath
  fi
  code=$(curl -gsLm30 -o /dev/null -w "%{http_code}" -I "$url") || code=000
  if [ "$code" -ge 400 ]; then
    code=$(curl -gsLm30 -o /dev/null -w "%{http_code}" -r 0-0 -A "$user_agent" "$url") || code=000
  fi
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
