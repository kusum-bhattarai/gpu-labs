#!/usr/bin/env bash
# PostToolUse hook: after an edit to a .cpp/.h file in 01-matmul, rebuild and run the correctness tests.
# Silent when everything passes. On failure, prints JSON that shows a message to the user (systemMessage)
# and sends the failure to Claude (decision=block + reason), so it's never silently ignored.
set -uo pipefail

file=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
case "$file" in
  */01-matmul/*.cpp|*/01-matmul/*.h) ;;
  *) exit 0 ;;
esac
case "$file" in */01-matmul/build/*|*/01-matmul/.venv/*) exit 0 ;; esac

cd "$(dirname "$0")/.." || exit 0
name=$(basename "$file")

report_fail() {  # $1 = short user message, $2 = details for Claude
  jq -n --arg msg "$1" --arg details "$2" '{
    systemMessage: $msg,
    decision: "block",
    reason: ($msg + "\n\n" + $details),
    hookSpecificOutput: { hookEventName: "PostToolUse", additionalContext: $details }
  }'
  exit 0
}

[[ -d build ]] || cmake -S . -B build >/dev/null 2>&1

if ! build_out=$(cmake --build build -j 2>&1); then
  errs=$(grep -E 'error' <<<"$build_out" | head -15)
  report_fail "✗ 01-matmul BUILD FAILED after editing $name" "${errs:-$build_out}"
fi

if ! test_out=$(./build/test_matmul 2>&1); then
  fails=$(grep -E 'FAIL' <<<"$test_out" | head -15)
  report_fail "✗ 01-matmul CORRECTNESS FAILED after editing $name" "$fails"
fi
exit 0
