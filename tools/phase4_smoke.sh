#!/usr/bin/env bash
# The Phase 4 gate for the two command-line tools: start imgjit-server on an ephemeral
# port, drive it with a pipelining imgjit-client asking for BOTH result paths, and check
# the server shut down cleanly on SIGINT.
#
# The unit suite already covers the protocol and the threading; what this adds is that
# the tools themselves work as advertised, including the ephemeral-port handshake that
# lets the suite run servers without picking a fixed port.
set -euo pipefail

server_bin=$1
client_bin=$2
input_png=$3
work_dir=$4

rm -rf "$work_dir"
mkdir -p "$work_dir/out"
log="$work_dir/server.log"

"$server_bin" --port 0 --slots 4 --slots-per-conn 2 --queue 4 --output-dir "$work_dir/out" \
  >"$log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

port=""
for _ in $(seq 1 100); do
  port=$(sed -n 's/.*127\.0\.0\.1:\([0-9][0-9]*\).*/\1/p' "$log" 2>/dev/null | head -n 1)
  [ -n "$port" ] && break
  sleep 0.1
done
if [ -z "$port" ]; then
  echo "phase4_smoke: the server never reported a listening port" >&2
  cat "$log" >&2
  exit 1
fi

# --repeat 8 against a 4-slot pool: every frame is sent before any response is read, so
# the reader parks on slot claim. That is the pipelining case decision 6 is about.
"$client_bin" --port "$port" --ops "grayscale,sobel" --repeat 8 --server-write \
  "$input_png" "$work_dir/echoed.png"

written=$(find "$work_dir/out" -name 'frame_*.png' | wc -l | tr -d ' ')
if [ "$written" -ne 8 ]; then
  echo "phase4_smoke: expected 8 server-side PNGs, found $written" >&2
  exit 1
fi
if [ ! -s "$work_dir/echoed.png" ]; then
  echo "phase4_smoke: the client wrote no echoed result" >&2
  exit 1
fi

kill -INT "$server_pid"
trap - EXIT
wait "$server_pid"
grep -q "8 frames completed, 0 failed" "$log" || {
  echo "phase4_smoke: the server did not report 8 clean frames" >&2
  cat "$log" >&2
  exit 1
}
echo "phase4_smoke: ok — 8 frames, both result paths, clean shutdown"
