#!/usr/bin/env bash

# Script to install Git hooks from .githooks directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DIR="$(git rev-parse --git-dir)"
HOOKS_DIR="${GIT_DIR}/hooks"

echo "Installing Git hooks..."

# Install pre-commit hook
echo "📦 Installing pre-commit hook..."
cp "${SCRIPT_DIR}/pre-commit" "${HOOKS_DIR}/pre-commit"
chmod +x "${HOOKS_DIR}/pre-commit"
echo "✅ pre-commit hook installed"

echo ""
echo "🎉 Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will automatically update .ci/docker/ci_commit_pins/pytorch.txt"
echo "whenever you commit changes to torch_pin.py"
