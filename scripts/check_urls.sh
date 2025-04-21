#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

ua="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
accept_hdr="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
lang_hdr="en-US,en;q=0.9"
enc_hdr="gzip, deflate, br"
cache_hdr="no-cache"
conn_hdr="keep-alive"

url="https://wiki.mozilla.org/Abstract_Interpretation"

curl -s -o /dev/null -w '%{http_code}\n' \
  -I \
  -A "$ua" \
  -H "Accept: $accept_hdr" \
  -H "Accept-Language: $lang_hdr" \
  -H "Accept-Encoding: $enc_hdr" \
  -H "Connection: $conn_hdr" \
  -H "Cache-Control: $cache_hdr" \
  "$url"

curl -s -o /dev/null -w '%{http_code}\n' \
  --range 0-0 \
  -A "$ua" \
  -H "Accept: $accept_hdr" \
  -H "Accept-Language: $lang_hdr" \
  -H "Accept-Encoding: $enc_hdr" \
  -H "Connection: $conn_hdr" \
  -H "Cache-Control: $cache_hdr" \
  "$url"

wget --spider --server-response --user-agent="$ua" "$url" 2>&1 \
  | awk '/^  HTTP\// { print $2 }'

wget --spider --server-response --user-agent="$ua" --method=GET "$url" 2>&1 \
  | awk '/^  HTTP\// { print $2 }'

curl -s -o /dev/null -w '%{http_code}\n' \
  --http1.1 \
  -I \
  -A "$ua" \
  -H "Accept: $accept_hdr" \
  -H "Accept-Language: $lang_hdr" \
  -H "Accept-Encoding: $enc_hdr" \
  -H "Connection: $conn_hdr" \
  -H "Cache-Control: $cache_hdr" \
  "$url"

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
