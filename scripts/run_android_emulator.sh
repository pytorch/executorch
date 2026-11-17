#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
set +m
script_deadline=$((SECONDS + ${ANDROID_EMULATOR_SCRIPT_TIMEOUT:-5400}))
execution_deadline=$script_deadline
active_pid=
launch_in_progress=false
pending_exit_status=0
collecting_diagnostics=false

# This script is originally adopted from https://github.com/pytorch/pytorch/blob/main/android/run_tests.sh
ADB_PATH="$ANDROID_HOME/platform-tools/adb"

instrumentation_args=()
case "${ANDROID_TEST_SUITE:-all}" in
  all) ;;
  functional)
    instrumentation_args=(-e notClass org.pytorch.executorch.LlmPerformanceTest)
    ;;
  performance)
    instrumentation_args=(-e class org.pytorch.executorch.LlmPerformanceTest)
    ;;
  *)
    echo "Unknown Android test suite: $ANDROID_TEST_SUITE" >&2
    exit 2
    ;;
esac

check_execution_deadline() {
  if ((SECONDS >= script_deadline)); then
    echo "Android emulator script exhausted its execution budget" >&2
    exit 124
  fi
  if ((SECONDS >= execution_deadline)); then
    echo "Emulator framework did not become ready before the deadline" >&2
    exit 1
  fi
}

handle_signal() {
  if [ "$collecting_diagnostics" = true ]; then
    return 0
  fi
  if ((pending_exit_status == 0)); then
    pending_exit_status="$1"
  fi
  if [ "$launch_in_progress" != true ]; then
    exit "$pending_exit_status"
  fi
}

stop_active_command() {
  local wait_pending="${1:-true}"
  if [ -z "$active_pid" ]; then
    return 0
  fi
  if ! kill -TERM -- "-$active_pid" 2>/dev/null && [ "$wait_pending" = true ]; then
    kill -TERM "$active_pid" 2>/dev/null || true
  fi
  if kill -0 -- "-$active_pid" 2>/dev/null; then
    /bin/sleep 1
    kill -KILL -- "-$active_pid" 2>/dev/null || true
  fi
  wait "$active_pid" 2>/dev/null || true
  active_pid=
}

run_tracked_command() {
  local status=0
  launch_in_progress=true
  setsid -- "$@" &
  active_pid=$!
  launch_in_progress=false
  if ((pending_exit_status != 0)); then
    exit "$pending_exit_status"
  fi
  wait "$active_pid" || status=$?
  stop_active_command false
  return "$status"
}

adb_with_deadline() {
  local deadline="$1"
  shift
  local remaining=$((deadline - SECONDS))
  if ((remaining <= 0)); then
    echo "ADB diagnostic budget exhausted" >&2
    return 124
  fi
  run_tracked_command timeout --foreground --kill-after=1s "${remaining}s" \
    timeout --foreground --kill-after=1s "${ANDROID_ADB_TIMEOUT:-60s}" "$ADB_PATH" "$@"
}

adb_with_timeout() {
  check_execution_deadline
  local status=0
  adb_with_deadline "$execution_deadline" "$@" || status=$?
  check_execution_deadline
  return "$status"
}

sleep_with_deadline() {
  check_execution_deadline
  local delay="$1"
  local remaining=$((execution_deadline - SECONDS))
  if ((delay > remaining)); then
    delay=$remaining
  fi
  run_tracked_command sleep "$delay"
  check_execution_deadline
}

collect_diagnostics() {
  local status=$?
  collecting_diagnostics=true
  set +e
  set +x
  trap '' PIPE
  launch_in_progress=false
  pending_exit_status=0
  stop_active_command
  local diagnostic_deadline=$((SECONDS + ${ANDROID_DIAGNOSTIC_BUDGET_SECONDS:-120}))
  cat result.txt
  adb_with_deadline "$diagnostic_deadline" logcat -d -b all >logcat.txt 2>&1
  adb_with_deadline "$diagnostic_deadline" devices -l >devices.txt 2>&1
  adb_with_deadline "$diagnostic_deadline" shell getprop >emulator-properties.txt 2>&1
  adb_with_deadline "$diagnostic_deadline" shell df -h /data >device-storage.txt 2>&1
  if ((status != 0)); then
    {
      printf 'Android harness exit status: %s\n' "$status"
      echo 'Last instrumentation output:'
      tail -n 40 result.txt
      echo 'Crash, timeout, and process-kill evidence:'
      awk '
        /FATAL EXCEPTION|Fatal signal|Abort message|TestTimedOutException|Native test teardown refused|lowmemorykiller|lmkd|am_crash|am_kill|am_proc_died|tombstoned| DEBUG / { context = 30 }
        context > 0 { recent[lines % 120] = $0; context--; lines++ }
        END {
          first = lines > 120 ? lines - 120 : 0
          for (i = first; i < lines; i++) print recent[i % 120]
        }
      ' logcat.txt
    } >failure-summary.txt 2>&1
    cat failure-summary.txt
  fi
  exit "$status"
}

: >result.txt
: >failure-summary.txt
trap collect_diagnostics EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 131' QUIT
trap 'handle_signal 141' PIPE
trap 'handle_signal 143' TERM

adb_shell_with_retries() {
  local attempts="$1"
  shift

  for ((i = 1; i <= attempts; i++)); do
    if ANDROID_ADB_TIMEOUT=5s adb_with_timeout shell "$@"; then
      return 0
    fi
    sleep_with_deadline 1
    ANDROID_ADB_TIMEOUT=5s adb_with_timeout wait-for-device || true
  done

  return 1
}

ready_deadline=$((SECONDS + ${ANDROID_EMULATOR_READY_TIMEOUT:-900}))
if ((ready_deadline > script_deadline)); then
  ready_deadline=$script_deadline
fi
execution_deadline=$ready_deadline

echo "Waiting for emulator boot to complete"
adb_with_timeout wait-for-device

# PackageManager can answer before StorageManager is ready to install APKs.
# shellcheck disable=SC2016
until adb_with_timeout shell \
  'test "$(getprop sys.boot_completed)" = "1" && cmd package list packages >/dev/null && sm list-volumes all >/dev/null'; do
  sleep_with_deadline 5
  adb_with_timeout wait-for-device || true
done
execution_deadline=$script_deadline

echo "Unlock emulator and disable animations"
adb_shell_with_retries 5 input keyevent 82 || true
adb_shell_with_retries 5 settings put global window_animation_scale 0.0 || true
adb_shell_with_retries 5 settings put global transition_animation_scale 0.0 || true
adb_shell_with_retries 5 settings put global animator_duration_scale 0.0 || true

# The device will be created by ReactiveCircus/android-emulator-runner GHA
echo "List all running emulators"
adb_with_timeout devices

adb_with_timeout uninstall org.pytorch.executorch.test || true

APK_PATH=android-test-debug-androidTest.apk
if [ ! -s "$APK_PATH" ]; then
  echo "Missing or empty test APK: $APK_PATH" >&2
  exit 1
fi
printf 'Test APK size: %s bytes\n' "$(wc -c < "$APK_PATH")" >install-summary.txt
cat install-summary.txt
install_deadline=$((SECONDS + ${ANDROID_APK_INSTALL_BUDGET_SECONDS:-900}))
if ((install_deadline > script_deadline)); then
  install_deadline=$script_deadline
fi
install_ok=false
for ((i = 1; i <= 5; i++)); do
  install_remaining=$((install_deadline - SECONDS))
  if ((install_remaining <= 0)); then
    echo "APK installation exhausted its execution budget" | tee -a install-summary.txt >&2
    if ((SECONDS >= script_deadline)); then
      exit 124
    fi
    break
  fi
  attempt_started=$SECONDS
  if run_tracked_command timeout --foreground --kill-after=5s "${install_remaining}s" \
    timeout --foreground --kill-after=5s "${ANDROID_APK_INSTALL_TIMEOUT:-600s}" \
    "$ADB_PATH" install -r -t "$APK_PATH" >"install-attempt-${i}.txt" 2>&1; then
    install_status=0
    install_ok=true
  else
    install_status=$?
  fi
  cat "install-attempt-${i}.txt"
  printf 'APK install attempt %s: exit=%s elapsed=%ss\n' \
    "$i" "$install_status" "$((SECONDS - attempt_started))" | tee -a install-summary.txt
  check_execution_deadline
  if [ "$install_ok" = "true" ]; then
    break
  fi
  if ((i < 5 && SECONDS < install_deadline)); then
    retry_delay=$((install_deadline - SECONDS))
    if ((retry_delay > 10)); then
      retry_delay=10
    fi
    sleep_with_deadline "$retry_delay"
    reconnect_timeout=$((install_deadline - SECONDS))
    if ((reconnect_timeout > 5)); then
      reconnect_timeout=5
    fi
    if ((reconnect_timeout > 0)); then
      run_tracked_command timeout --foreground --kill-after=1s "${reconnect_timeout}s" "$ADB_PATH" wait-for-device || true
    fi
  fi
done
if [ "$install_ok" != "true" ]; then
  echo "Failed to install test APK after retries" >&2
  exit 1
fi

adb_with_timeout logcat -c || true
check_execution_deadline
remaining=$((script_deadline - SECONDS))
run_tracked_command timeout --foreground --kill-after=5s "${remaining}s" \
  timeout --foreground --kill-after=5s "${ANDROID_INSTRUMENTATION_TIMEOUT:-3600s}" \
  "$ADB_PATH" shell am instrument -w -r "${instrumentation_args[@]}" \
  org.pytorch.executorch.test/androidx.test.runner.AndroidJUnitRunner >result.txt 2>&1
check_execution_deadline

if grep -Eq 'FAILURES|INSTRUMENTATION_FAILED|INSTRUMENTATION_ABORTED|shortMsg=' result.txt; then
  echo "Android instrumentation reported a failure" >&2
  exit 1
fi
if ! grep -Eq '^OK \([1-9][0-9]* tests?\)[[:space:]]*$' result.txt || \
  [ "$(grep -c '^INSTRUMENTATION_CODE:' result.txt)" -ne 1 ] || \
  ! grep -Eq '^INSTRUMENTATION_CODE: -1[[:space:]]*$' result.txt; then
  echo "Android instrumentation did not report a completed, nonempty passing suite" >&2
  exit 1
fi
