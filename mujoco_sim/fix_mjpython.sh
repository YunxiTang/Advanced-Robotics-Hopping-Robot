#!/usr/bin/env bash
# Workaround for `mjpython` failing to dlopen libpython on macOS when the venv
# uses uv's python-build-standalone interpreter: mjpython's native trampoline
# resolves @executable_path against the *symlinked* .venv/bin/python path, so
# it looks for the dylib under .venv/lib/ instead of the real interpreter's
# lib/ directory. Symlinking it into place fixes this.
set -euo pipefail
cd "$(dirname "$0")"
REAL_PY="$(readlink -f .venv/bin/python)"
DYLIB="$(dirname "$(dirname "$REAL_PY")")/lib/libpython3.11.dylib"
if [ -f "$DYLIB" ]; then
  ln -sf "$DYLIB" .venv/lib/libpython3.11.dylib
  echo "Linked $DYLIB -> .venv/lib/libpython3.11.dylib"
else
  echo "libpython3.11.dylib not found at $DYLIB" >&2
  exit 1
fi
