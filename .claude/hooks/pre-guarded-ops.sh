#!/bin/bash
# Block "guarded" operations that must always be approved by the user,
# even in auto/bypass mode. PreToolUse(Bash) hooks still run under bypass
# permissions, so this is the reliable enforcement point.
# Guarded operations (per CLAUDE.md "Guarded Operations"):
#   1. pkill targeting (g)unicorn workers, e.g. `pkill -f unicorn`
#   2. setting nginx worker_processes to a manual count (not `auto`)
# To proceed after the user has explicitly approved, prefix the command
# with the acknowledgement marker GUARDED_OPS_ACK=1.
set -euo pipefail

input=$(cat)
command=$(
    echo "$input" \
        | jq -r '.tool_input.command // empty'
)

if [[ -z "$command" ]]; then
    exit 0
fi

# Explicit, user-approved override. The marker must only be added after
# the user grants permission (see CLAUDE.md).
if [[ "$command" == *"GUARDED_OPS_ACK=1"* ]]; then
    exit 0
fi

reason=""

# 1) pkill of (g)unicorn workers.
if echo "$command" | grep -Eqi 'pkill[^|;&]*unicorn'; then
    reason="pkill targeting (g)unicorn workers"
fi

# 2) nginx worker_processes set to a manual (numeric) value. Commands that
#    set the value via sed/tee/echo include the number, so this catches
#    them; read-only commands (grep/cat) do not match and are allowed.
#    Edits made through the Edit/Write tool are covered behaviorally by the
#    CLAUDE.md rule.
if echo "$command" | grep -Eqi 'worker_processes[[:space:]]*[0-9]'; then
    reason="nginx worker_processes set to a manual count"
fi

if [[ -n "$reason" ]]; then
    cat >&2 <<EOF
BLOCKED (guarded operation): $reason
This operation can disrupt running services and is guarded by CLAUDE.md.
You MUST warn the user and obtain explicit approval before running it,
even in auto mode. After the user approves, re-run the command prefixed
with GUARDED_OPS_ACK=1 to proceed.
EOF
    exit 2
fi

exit 0
