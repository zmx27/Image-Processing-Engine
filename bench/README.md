# Benchmarks

`imgjit-bench` is the timing harness introduced in Phase 5 (`docs/PLAN.md`). It drives a running
`imgjit-server` over a socket with `--connections` concurrent clients, each holding `--window`
frames in flight, and reports FPS plus p50/p99 round-trip latency. Phase 8 widens the same binary
over the naive|tiled × sync|async matrix rather than starting a second one, so every number Phases
6 and 7 are gated against was produced by the same instrument.

## Why the numbers here matter

Phases 6 and 7 are both gated on being *measurably faster*. That is only a claim if something
recorded a number first, under a harness that still exists when the comparison is made. This
directory is that record: `baseline_phase5.csv` is the single-stream, synchronous, naive-kernel
starting point, and every later row is compared against it like-for-like.

## Recording the Phase 5 baseline

Needs a real NVIDIA GPU, so this runs on Colab (`colab/run.ipynb` has it as a cell). Two runs,
differing only in whether the chain was prewarmed:

    ./build/tools/imgjit-server --port 9000 --backend cuda \
        --slots 8 --slots-per-conn 2 --queue 16 --max-payload 16777216 &

    ./build/bench/imgjit-bench --port 9000 --connections 4 --frames 200 \
        --width 1024 --height 1024 --channels 3 \
        --ops "grayscale,gaussian:1.4,sobel,threshold:0.3" \
        --label phase5_cuda_cold --csv bench/baseline_phase5.csv

Then restart the server with `--prewarm "grayscale,gaussian:1.4,sobel,threshold:0.3"` and rerun
with `--label phase5_cuda_prewarmed`. The two rows differ in `cold_first_ms` and nowhere else,
which is what turns `docs/ARCHITECTURE.md`'s "NVRTC cold-start is ~50–200 ms, exactly why the cache
exists" from an assertion into a measurement.

A CPU-backend row (`--backend cpu`, no prewarm) is worth recording alongside them. It is not a
fair GPU comparison — it is the scalar oracle — but it is the only row that can be reproduced on a
Mac, which makes it the one that catches a harness regression without a GPU.

## Recording the Phase 6 comparison

Phase 6 made the pipeline asynchronous across K streams. The A/B is **two runs of the same
server binary**, differing only in `--streams`, so nothing about the comparison depends on a build
that no longer exists:

    # the Phase 5 pipeline: one frame on the GPU at a time
    ./build/tools/imgjit-server --port 9000 --backend cuda --streams 1 \
        --slots 8 --slots-per-conn 2 --queue 16 --max-payload 16777216 \
        --prewarm "grayscale,gaussian:1.4,sobel,threshold:0.3" &

    ./build/bench/imgjit-bench --port 9000 --connections 4 --frames 200 \
        --width 1024 --height 1024 --channels 3 \
        --ops "grayscale,gaussian:1.4,sobel,threshold:0.3" \
        --label phase6_streams1 --csv bench/baseline_phase6.csv

Then restart with `--streams 4` and rerun with `--label phase6_streams4`. Same slots, same queue,
same payload, same chain — only the number of frames the GPU may hold at once changes.

**Read the server's shutdown line as well as the CSV.** It reports mean and peak stream occupancy,
and that is the half of the phase gate throughput cannot answer: a run whose `mean in flight` sits
near 1.0 under `--streams 4` did not overlap anything, and whatever made it faster was not the
pipeline. Confirm the same thing visually with Nsight Systems, which is the other Phase 6 gate:

    nsys profile -o phase6 --trace=cuda ./build/tools/imgjit-server --port 9000 \
        --backend cuda --streams 4 ...

The timeline must show H2D, kernel and D2H rows genuinely interleaved across streams, not a single
file of segments with gaps between them.

## Reading the columns

| Column | Meaning |
|---|---|
| `fps` | total frames ÷ wall time, across all connections |
| `req_mb_per_s` | request payload only; the echo doubles the bytes actually on the wire |
| `p50_ms` / `p99_ms` / `mean_ms` / `max_ms` | round trip, from `send()` entry to matched response |
| `cold_first_ms` | the worst first frame across connections, excluded from the percentiles |
| `failures` | frames answered with a non-zero status; a baseline row must have 0 |

`cold_first_ms` is reported separately rather than folded in because on the CUDA backend that frame
pays the NVRTC compile unless the chain was prewarmed — averaging a compile into a run that
measures execution would put a 100 ms outlier in the p99 of every cold run and hide the effect the
cache exists to produce.

Round trip is measured per `seq_num`, never by arrival order: under pipelining the Nth response is
not the Nth request, which is the entire reason `seq_num` is in the protocol
(`docs/PROTOCOL.md`). The timestamp lives on the client's pending map
(`include/imgjit/net/client.h`) for that reason.
