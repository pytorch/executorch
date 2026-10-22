#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# test_multisession.sh
#
# Black-box smoke test for the muse_glimmer OpenAI-compatible server's multi-session
# support. It exercises three independent properties over plain HTTP (no code
# imports, no access to the server process) so it can run against any live
# server instance:
#
#   TEST 1  Same-session multi-turn context coherence
#           Two turns on the SAME session_id. The client sends the full
#           conversation history on turn 2 and the model must recall a secret
#           code introduced on turn 1. This proves the server correctly threads
#           the client-supplied history through a multi-turn exchange.
#
#           IMPORTANT: this does NOT prove KV warm-resume was engaged. The
#           warm-resume optimization is output-transparent by design -- with or
#           without it, and regardless of session_id, the reply is identical
#           because correctness comes from the history in the REQUEST, not from
#           server-side memory. The only signal that KV was actually reused is
#           the server-side log line
#             llm_turn_stats ... reused_prompt_tokens=<N>
#           which is NOT exposed in the HTTP response, so it cannot be asserted
#           from this black-box script. To confirm reuse, watch the server
#           stdout for reused_prompt_tokens > 0 on the second turn.
#
#   TEST 2  Concurrent session isolation
#           N distinct session_ids fired in parallel, each asked to echo its own
#           unique token. Every response must contain ONLY its own token ->
#           proves concurrent sessions do not cross-contaminate.
#
#   TEST 3  Cross-session isolation
#           A fresh session asks for another session's secret. It must NOT know
#           it -> proves per-session state is isolated.
#
# The server caps live sessions at --max-sessions (default 16). This script
# tracks every session_id it opens and closes them all on exit (DELETE
# /v1/sessions/{id}), so it is self-cleaning and safe to re-run back-to-back
# without exhausting the session pool.
#
# The server must be started separately (this script does NOT launch it). Start
# it, for example, with:
#
#   PYTHONPATH=/path/to/executorch:/path/to/parent \
#   LD_LIBRARY_PATH=<cmake-out>/lib:<cmake-out>/lib64 \
#   CUDA_VISIBLE_DEVICES=0 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
#   conda run -n <env> --no-capture-output python -m \
#     executorch.examples.models.muse_glimmer.serving.serve \
#     --model-path <model.pte> --data-path <blob.ptd> \
#     --tokenizer-path <tokenizer.json> --hf-tokenizer <hf_dir> \
#     --worker-bin <cmake-out>/examples/models/muse-glimmer/muse_glimmer_worker \
#     --tool-parser atem --model-id muse_glimmer --max-context 131072 \
#     --max-sessions 16 --host 127.0.0.1 --port 8000
#
# -----------------------------------------------------------------------------
# USAGE
#   bash test_multisession.sh [HOST_PORT] [MODEL_ID] [NUM_CONCURRENT]
#
#   HOST_PORT        host:port of the running server   (default: 127.0.0.1:8000)
#   MODEL_ID         model id served by the server     (default: muse_glimmer)
#   NUM_CONCURRENT   concurrent sessions for TEST 2     (default: 8)
#
# EXAMPLES
#   bash test_multisession.sh
#   bash test_multisession.sh 127.0.0.1:8001
#   bash test_multisession.sh 127.0.0.1:8000 muse_glimmer 16
#
# REQUIREMENTS
#   curl, python3 (stdlib only). No extra packages.
#
# EXIT CODE
#   0 if all three tests pass, non-zero otherwise.
# =============================================================================

set -u

HOST_PORT="${1:-127.0.0.1:8000}"
MODEL_ID="${2:-muse_glimmer}"
NUM_CONCURRENT="${3:-8}"
BASE="http://${HOST_PORT}/v1/chat/completions"
HEALTH="http://${HOST_PORT}/health"

# Never route localhost through a proxy.
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

CURL=(curl -s --noproxy '*' -m 120 -H "Content-Type: application/json")
SESS_BASE="http://${HOST_PORT}/v1/sessions"

# Track every session_id we create so we can free them on exit. The server
# caps live sessions at --max-sessions; leaking them would exhaust the pool and
# make later runs fail with "capacity_exhausted".
OPENED_SESSIONS=()
track_session() { OPENED_SESSIONS+=("$1"); }
cleanup_sessions() {
  for sid in "${OPENED_SESSIONS[@]:-}"; do
    [ -n "$sid" ] || continue
    curl -s --noproxy '*' -m 15 -X DELETE "${SESS_BASE}/${sid}" -o /dev/null 2>/dev/null || true
  done
}
trap cleanup_sessions EXIT

pass=0
fail=0
ok()   { echo "  [PASS] $1"; pass=$((pass + 1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail + 1)); }

# Extract the assistant message content from a chat-completion response.
content_of() {
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content') or '')"
}

# -----------------------------------------------------------------------------
echo "=== muse_glimmer multi-session smoke test ==="
echo "server : ${BASE}"
echo "model  : ${MODEL_ID}"
echo

# Health check first so failures are obvious.
if ! "${CURL[@]}" -o /dev/null -w '%{http_code}' "${HEALTH}" 2>/dev/null | grep -q '200'; then
  echo "ERROR: server health check failed at ${HEALTH}"
  echo "       Start the server first (see header of this script), then retry."
  exit 2
fi
echo "health : ok"
echo

# -----------------------------------------------------------------------------
# TEST 1: same-session multi-turn context (recall a secret code).
# -----------------------------------------------------------------------------
echo "TEST 1: same-session multi-turn context coherence"
# NOTE: This checks that the model consumes the client-supplied history, not
# that KV warm-resume fired (warm-resume is output-transparent; its only signal
# is the server-side reused_prompt_tokens log, not visible over HTTP).
SESS_TRIP="sessTrip_$"
track_session "$SESS_TRIP"
SECRET="FALCON-77"
LONG="You are helping me plan a trip. I have 7 days and a budget of 3000 dollars, \
I love hiking and museums, and I want you to remember this secret trip code: ${SECRET}. \
Please acknowledge with OK and the code."

REQ1=$(python3 -c '
import json,sys
long=sys.argv[1]; sid=sys.argv[2]; model=sys.argv[3]
print(json.dumps({"model":model,"session_id":sid,"temperature":0,"max_tokens":40,
                  "messages":[{"role":"user","content":long}]}))
' "$LONG" "$SESS_TRIP" "$MODEL_ID")
A1=$(printf '%s' "$REQ1" | "${CURL[@]}" -d @- "$BASE" | content_of)
echo "  turn1 assistant: $(printf '%s' "$A1" | tr '\n' ' ' | cut -c1-80)"

REQ2=$(python3 -c '
import json,sys
long=sys.argv[1]; a1=sys.argv[2]; sid=sys.argv[3]; model=sys.argv[4]
msgs=[{"role":"user","content":long},
      {"role":"assistant","content":a1},
      {"role":"user","content":"What was the secret trip code I gave you? Answer with just the code."}]
print(json.dumps({"model":model,"session_id":sid,"temperature":0,"max_tokens":40,"messages":msgs}))
' "$LONG" "$A1" "$SESS_TRIP" "$MODEL_ID")
A2=$(printf '%s' "$REQ2" | "${CURL[@]}" -d @- "$BASE" | content_of)
echo "  turn2 assistant: $(printf '%s' "$A2" | tr '\n' ' ' | cut -c1-80)"

if printf '%s' "$A2" | grep -q "$SECRET"; then
  ok "turn2 recalled the secret code (${SECRET}) from the supplied history"
else
  bad "turn2 did NOT recall the secret code (${SECRET})"
fi
echo

# -----------------------------------------------------------------------------
# TEST 2: concurrent session isolation.
# -----------------------------------------------------------------------------
echo "TEST 2: ${NUM_CONCURRENT} concurrent distinct sessions must not cross"
TMPDIR_T2=$(mktemp -d)
CONCUR_PREFIX="concur_$"
for i in $(seq 1 "$NUM_CONCURRENT"); do
  track_session "${CONCUR_PREFIX}_${i}"
done
for i in $(seq 1 "$NUM_CONCURRENT"); do
  (
    REQ=$(python3 -c '
import json,sys
i=sys.argv[1]; model=sys.argv[2]; sid=sys.argv[3]
print(json.dumps({"model":model,"session_id":sid,
                  "temperature":0,"max_tokens":16,
                  "messages":[{"role":"user","content":f"Reply with exactly this word and nothing else: TOKEN{i}"}]}))
' "$i" "$MODEL_ID" "${CONCUR_PREFIX}_${i}")
    printf '%s' "$REQ" | "${CURL[@]}" -d @- "$BASE" | content_of > "${TMPDIR_T2}/${i}.out"
  ) &
done
wait

cross=0
for i in $(seq 1 "$NUM_CONCURRENT"); do
  out=$(cat "${TMPDIR_T2}/${i}.out" 2>/dev/null)
  if ! printf '%s' "$out" | grep -q "TOKEN${i}\b"; then
    echo "    session ${i}: expected TOKEN${i}, got: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-40)"
    cross=$((cross + 1))
  fi
  # Also verify no OTHER session's token leaked in.
  for j in $(seq 1 "$NUM_CONCURRENT"); do
    if [ "$j" -ne "$i" ] && printf '%s' "$out" | grep -q "TOKEN${j}\b"; then
      echo "    session ${i}: LEAKED TOKEN${j}"
      cross=$((cross + 1))
    fi
  done
done
rm -rf "$TMPDIR_T2"

if [ "$cross" -eq 0 ]; then
  ok "all ${NUM_CONCURRENT} concurrent sessions returned only their own token"
else
  bad "${cross} concurrent-session mismatch/leak(s) detected"
fi
echo

# -----------------------------------------------------------------------------
# TEST 3: cross-session isolation (a fresh session must not know the secret).
# -----------------------------------------------------------------------------
echo "TEST 3: fresh session must NOT know another session's secret"
SESS_OTHER="sessOther_$"
track_session "$SESS_OTHER"
REQ3=$(python3 -c '
import json,sys
sid=sys.argv[1]; model=sys.argv[2]
print(json.dumps({"model":model,"session_id":sid,"temperature":0,"max_tokens":40,
                  "messages":[{"role":"user","content":"What was the secret trip code I gave you earlier? If you were not given one, say NONE."}]}))
' "$SESS_OTHER" "$MODEL_ID")
A3=$(printf '%s' "$REQ3" | "${CURL[@]}" -d @- "$BASE" | content_of)
echo "  assistant: $(printf '%s' "$A3" | tr '\n' ' ' | cut -c1-80)"

if printf '%s' "$A3" | grep -q "$SECRET"; then
  bad "fresh session leaked the other session's secret (${SECRET})"
else
  ok "fresh session did not know the other session's secret"
fi
echo

# -----------------------------------------------------------------------------
echo "=== summary: ${pass} passed, ${fail} failed ==="
[ "$fail" -eq 0 ]
