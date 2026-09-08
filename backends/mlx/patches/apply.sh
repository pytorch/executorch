#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Safely migrate and idempotently apply local patches to the MLX submodule.
#
# Usage:
#   apply.sh <mlx_source_dir> [--migrate <old> <current>]
#            [--remove <obsolete>] [--apply <patch>] ...
#   apply.sh <mlx_source_dir> <patch> [<patch> ...]
#
# Bare patch arguments preserve the original idempotent apply interface.
# Migrations and removals recognize only exact patch states. If a patch is
# partially applied or overlapping states are recognizable, the operation fails.
# Completed operations are rolled back if a later operation fails.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  apply.sh <mlx_source_dir> [--migrate <old> <current>]
           [--remove <obsolete>] [--apply <patch>] ...
  apply.sh <mlx_source_dir> <patch> [<patch> ...]
EOF
  exit 2
}

if (( $# < 1 )); then
  usage
fi

mlx_dir="$1"
shift

if ! git -C "$mlx_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a Git work tree: $mlx_dir" >&2
  exit 2
fi

rollback_actions=()
rollback_patches=()
transaction_active=1

rollback() {
  local status=$?
  local i

  if (( transaction_active == 0 || ${#rollback_actions[@]} == 0 )); then
    return "$status"
  fi

  echo "Patch operation failed; restoring the original patch state." >&2
  set +e
  for ((i = ${#rollback_actions[@]} - 1; i >= 0; i--)); do
    if [[ "${rollback_actions[$i]}" == "apply" ]]; then
      git -C "$mlx_dir" apply --verbose "${rollback_patches[$i]}"
    else
      git -C "$mlx_dir" apply --reverse --verbose "${rollback_patches[$i]}"
    fi
    if (( $? != 0 )); then
      echo "Failed to roll back patch: ${rollback_patches[$i]}" >&2
    fi
  done
  return "$status"
}
trap rollback EXIT

normalize_patch_path() {
  local patch="$1"
  local patch_dir
  local patch_name

  if [[ ! -f "$patch" ]]; then
    echo "Patch file does not exist: $patch" >&2
    return 1
  fi
  patch_dir="$(cd "$(dirname "$patch")" && pwd)"
  patch_name="$(basename "$patch")"
  printf '%s/%s\n' "$patch_dir" "$patch_name"
}

can_apply() {
  git -C "$mlx_dir" apply --check "$1" >/dev/null 2>&1
}

can_reverse() {
  git -C "$mlx_dir" apply --reverse --check "$1" >/dev/null 2>&1
}

apply_patch() {
  local patch="$1"
  echo "Applying MLX patch: $patch"
  git -C "$mlx_dir" apply --verbose "$patch"
  rollback_actions+=("reverse")
  rollback_patches+=("$patch")
}

reverse_patch() {
  local patch="$1"
  echo "Reversing MLX patch: $patch"
  git -C "$mlx_dir" apply --reverse --verbose "$patch"
  rollback_actions+=("apply")
  rollback_patches+=("$patch")
}

apply_idempotently() {
  local patch
  patch="$(normalize_patch_path "$1")"
  local forward=0
  local reverse=0
  can_apply "$patch" && forward=1
  can_reverse "$patch" && reverse=1

  if (( forward == 1 && reverse == 1 )); then
    echo "Ambiguous patch state (both apply and reverse checks pass): $patch" >&2
    return 1
  elif (( reverse == 1 )); then
    echo "MLX patch already applied, skipping: $patch"
  elif (( forward == 1 )); then
    apply_patch "$patch"
  else
    echo "Patch is neither cleanly applicable nor exactly applied: $patch" >&2
    return 1
  fi
}

remove_idempotently() {
  local patch
  patch="$(normalize_patch_path "$1")"
  local forward=0
  local reverse=0

  can_apply "$patch" && forward=1
  can_reverse "$patch" && reverse=1

  if (( forward == 1 && reverse == 1 )); then
    echo "Ambiguous obsolete patch state (both checks pass): $patch" >&2
    return 1
  elif (( reverse == 1 )); then
    reverse_patch "$patch"
  elif (( forward == 1 )); then
    echo "Obsolete MLX patch is not applied, skipping: $patch"
  else
    echo "Obsolete patch is neither absent nor exactly applied: $patch" >&2
    return 1
  fi
}

migrate_patch() {
  local old_patch
  old_patch="$(normalize_patch_path "$1")"
  local current_patch
  current_patch="$(normalize_patch_path "$2")"
  local old_forward=0
  local old_reverse=0
  local current_forward=0
  local current_reverse=0

  can_apply "$old_patch" && old_forward=1
  can_reverse "$old_patch" && old_reverse=1
  can_apply "$current_patch" && current_forward=1
  can_reverse "$current_patch" && current_reverse=1

  if (( old_reverse == 1 && current_reverse == 1 )); then
    echo "Ambiguous migration state (both old and current patches appear applied)." >&2
    echo "Old patch: $old_patch" >&2
    echo "Current patch: $current_patch" >&2
    return 1
  elif (( current_reverse == 1 )); then
    if (( current_forward == 1 )); then
      echo "Ambiguous current patch state (both checks pass): $current_patch" >&2
      return 1
    fi
    echo "Current MLX patch already applied, skipping: $current_patch"
  elif (( old_reverse == 1 )); then
    if (( old_forward == 1 )); then
      echo "Ambiguous legacy patch state (both checks pass): $old_patch" >&2
      return 1
    fi
    reverse_patch "$old_patch"
    if ! can_apply "$current_patch" || can_reverse "$current_patch"; then
      echo "Legacy patch reversed, but current patch state is ambiguous or conflicting: $current_patch" >&2
      return 1
    fi
    apply_patch "$current_patch"
  elif (( current_forward == 1 )); then
    # On a clean source tree both revisions may apply because they replace the
    # same upstream lines. The exact current forward-check makes this safe.
    apply_patch "$current_patch"
  else
    echo "Cannot identify a safe legacy, current, or clean migration state." >&2
    echo "Old patch: $old_patch" >&2
    echo "Current patch: $current_patch" >&2
    return 1
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --migrate)
      (( $# >= 3 )) || usage
      migrate_patch "$2" "$3"
      shift 3
      ;;
    --remove)
      (( $# >= 2 )) || usage
      remove_idempotently "$2"
      shift 2
      ;;
    --apply)
      (( $# >= 2 )) || usage
      apply_idempotently "$2"
      shift 2
      ;;
    --)
      shift
      while (( $# > 0 )); do
        apply_idempotently "$1"
        shift
      done
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      apply_idempotently "$1"
      shift
      ;;
  esac
done

transaction_active=0
