#!/bin/zsh
# Launch a sweep OUT of macOS background QoS. A process started from a
# background shell is given low-priority QoS ("SN" in ps) and runs ~2x slower;
# taskpolicy -B removes that after launch.
cd "$(dirname "$0")"
PY=/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python3
LOG="bdh_cl/results/logs/$2.log"
mkdir -p bdh_cl/results/logs
caffeinate -dimsu $PY -u bdh_cl/scripts/run_cl.py --config "$1" > "$LOG" 2>&1 &
PID=$!
sleep 6
taskpolicy -B -p $PID 2>/dev/null && echo "unthrottled pid=$PID" || echo "taskpolicy failed pid=$PID"
echo "log: $LOG"
