#!/usr/bin/env bash
# 运行 MyAgent 评测并与上一次结果对比。
# 用法: ./eval.sh [-k 用例过滤] [--label 标签]
set -uo pipefail
cd "$(dirname "$0")"

PY="$PWD/.venv/bin/python"
DEEPEVAL="$PWD/.venv/bin/deepeval"
export PATH="$PWD/.venv/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "评测要求 Python 3.12"' || exit 1
export DEEPEVAL_RESULTS_FOLDER="${DEEPEVAL_RESULTS_FOLDER:-$PWD/evals/results}"
export DEEPEVAL_TELEMETRY_OPT_OUT=1

PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) export MYAGENT_EVAL_LABEL="$2"; shift 2 ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

DEEPEVAL_RESULTS_FOLDER="$DEEPEVAL_RESULTS_FOLDER/features" "$DEEPEVAL" test run evals/test_features.py --verbose || exit 1

"$DEEPEVAL" test run evals/test_agent.py --verbose "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
status=$?

echo
"$PY" -m evals.compare --fail-on-regression
compare_status=$?

if [[ $status -ne 0 || $compare_status -ne 0 ]]; then
  exit 1
fi
