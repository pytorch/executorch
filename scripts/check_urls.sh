#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
URL="https://wiki.mozilla.org/Abstract_Interpretation"
PREFIX=$(hostname)_$(date +%s)

dig +short "$URL" | tee "${PREFIX}_dns.txt"

openssl s_client \
  -connect wiki.mozilla.org:443 \
  -servername wiki.mozilla.org \
  -alpn h2 < /dev/null \
  2>&1 | tee "${PREFIX}_tls.txt"

curl -vvv -A "$UA" -I "$URL" 2>&1 | tee "${PREFIX}_head_http2.txt"

curl -vvv --http1.1 -A "$UA" -I "$URL" 2>&1 | tee "${PREFIX}_head_http1.txt"

curl -vvv -A "$UA" --range 0-0 "$URL" 2>&1 | tee "${PREFIX}_range_http2.txt"

curl -vvv --http1.1 -A "$UA" --range 0-0 "$URL" 2>&1 | tee "${PREFIX}_range_http1.txt"

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
