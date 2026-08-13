#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

trap 'kill 0' SIGINT

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
egress_probe_url="https://api.github.com"
max_jobs=10
pids=()

running_jobs() {
  jobs -rp | wc -l
}

while IFS=: read -r filepath url; do
  (
    code=$(curl -k -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -I "$url") || code=000
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      sleep 1
      code=$(curl -k -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -r 0-0 -A "$user_agent" "$url") || code=000
    fi
    if [[ "$code" == "000" ]]; then
      # No HTTP response at all. Usually a dead host, but a runner with no egress
      # gets 000 for everything, and failing 2000 live links helps nobody. One
      # probe of a host the runner already depends on tells the two apart.
      probe=$(curl -k -gsLm10 -o /dev/null -w "%{http_code}" "$egress_probe_url") || probe=000
      if [[ "$probe" == "000" ]]; then
        printf "${yellow}WARN %s${reset} ${cyan}%s${reset} %s\n" "$code" "$url" "$filepath"
        exit 0
      fi
      # Egress works, so the host is either dead or slow. Give it one more try
      # with a longer timeout before failing it.
      code=$(curl -k -gsLm60 -o /dev/null -w "%{http_code}" -A "$user_agent" "$url") || code=000
    fi
    # Treat Cloudflare JS-challenge and rate-limit as success.
    if [[ "$code" == "403" || "$code" == "429" || "$code" == "503" ]]; then
      printf "${yellow}WARN %s${reset} ${cyan}%s${reset} %s\n" "$code" "$url" "$filepath"
      exit 0
    fi
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      printf "${red}FAIL %s${reset} ${yellow}%s${reset} %s\n" "$code" "$url" "$filepath" >&2
      exit 1
    else
      printf "${green} OK  %s${reset} ${cyan}%s${reset} %s\n" "$code" "$url" "$filepath"
      exit 0
    fi
  ) &
  pids+=($!)
  while [ "$(running_jobs)" -ge "$max_jobs" ]; do
    sleep 1
  done
done < <(
  pattern='(?!.*@lint-ignore)(?<!git\+)(?<!\$\{)https?://(?![^/]*@)(?![^\s<>\")]*[<>\{\}\$])[^[:space:]<>")\[\]\\|]+'
  excludes=(
    ':(exclude,glob)**/.*'
    ':(exclude,glob)**/*.lock'
    ':(exclude,glob)**/*.svg'
    ':(exclude,glob)**/*.xml'
    ':(exclude,glob)**/*.gradle*'
    ':(exclude,glob)**/*gradle*'
    ':(exclude,glob)**/third-party/**'
    ':(exclude,glob)**/third_party/**'
  )
  if [ $# -eq 2 ]; then
    # Three dots, not two. Against the base branch tip, a branch cut before recent
    # commits looks like it is adding back every line those commits touched.
    for filename in $(git diff --no-color --name-only --unified=0 "$1...$2"); do
      git diff --no-color --unified=0 "$1...$2" -- "$filename" "${excludes[@]}" \
        | grep -E '^\+' \
        | grep -Ev '^\+\+\+' \
        | perl -nle 'print for m#'"$pattern"'#g' \
        | sed 's|^|'"$filename"':|'
    done
  else
    git --no-pager grep --no-color -I -P -o "$pattern" -- . "${excludes[@]}"
  fi \
  | sed -E 's/[^/[:alnum:]]+$//' \
  | grep -Ev '://(0\.0\.0\.0|127\.0\.0\.1|localhost)([:/])' \
  | grep -Ev '://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' \
  | grep -Ev 'fwdproxy:8080' \
  || true
)

for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || {
    case $? in
      1) status=1 ;;
      127) ;;  # ignore "not a child" noise
      *) exit $? ;;
    esac
  }
done

exit $status
