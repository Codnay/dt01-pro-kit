#!/usr/bin/env bash
# Run the unit tests. No robot is needed and nothing moves.
#
#   scripts/run_tests.sh                    everything
#   scripts/run_tests.sh test_ik_solver*    only the matching arm tests
#
# Each test file runs in its own process: several of them replace heavy
# libraries with small stand-ins, which must not leak into the next file.
#
# The arm tests were written against generic settings, so they run against
# tests/fixtures/generic_configs.py rather than this robot's tuning.
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$KIT/.venv/bin/python}"
cd "$KIT"

# One panel test loads the default jobs file, which creates ops/jobs.json from
# the template when it is missing. Do not leave that behind: setup.sh must be
# the one to write it, with this machine's real path.
if [[ ! -e ops/jobs.json ]]; then
  trap 'rm -f "$KIT/ops/jobs.json"' EXIT
fi

failed=0
run() {  # run <label> <command...>
  local label="$1"; shift
  if output="$("$@" 2>&1)"; then
    printf '  ok    %s  (%s)\n' "$label" "$(grep -E '^Ran ' <<<"$output" | tail -1)"
  else
    printf '  FAIL  %s\n' "$label"
    tail -25 <<<"$output" | sed 's/^/        /'
    failed=1
  fi
}

echo "== arms and control panel"
for file in tests/${1:-test_}*.py; do
  run "$(basename "$file")" env \
    PIPER_CONFIG_PATH="$KIT/tests/fixtures/generic_configs.py" PYTHONPATH="$KIT" \
    "$PY" -m unittest discover -s tests -p "$(basename "$file")"
done

if [[ $# -eq 0 ]]; then
  echo "== base and lift dashboard"
  run "base/dashboard/tests" python3 -m unittest discover -s base/dashboard/tests
fi

[[ "$failed" == 0 ]] && echo "All tests passed." || echo "Some tests FAILED."
exit "$failed"
