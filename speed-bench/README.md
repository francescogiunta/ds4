## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 512 \
  --gen-warmup-tokens 32
```

`--gen-warmup-tokens` does not generate extra tokens. It appends warmup and
post-warmup fields to the CSV while preserving the meaning and order of the
existing columns. With the values above, `gen_measured_tps` covers the final
480 tokens. The option defaults to 0.

For the GB10 65K/100K campaign, use
`tests/long_context_story_prompt_gb10.txt`. It is generated deterministically
from the canonical fact-recall story by
`tests/generate_gb10_long_context_fixture.py` and retains one final query.

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Memory guard

On Linux, run model-backed benchmarks through `memory_guard.py`. It samples the
whole child process group every 250 ms, records RSS, periodically records PSS,
queries NVIDIA per-process accounting every second, tracks host `MemAvailable`,
and writes both a CSV trace and a JSON summary. NVIDIA memory may be reported as
unavailable on unified-memory systems; the host and process limits remain
active. The defaults stop the run at 110 GB decimal group RSS/PSS, leaving a
5 GB margin below the campaign's 115 GB hard cap. The same default cap is
enforced against NVIDIA per-process memory when that metric is available. The
guard also stops when host available memory falls to 12 GiB:

```
python3 speed-bench/memory_guard.py \
  --csv speed-bench/local-runs/q2-65k.memory.csv \
  --summary-json speed-bench/local-runs/q2-65k.memory.json \
  -- \
  ./ds4-bench \
    -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 65536 --ctx-max 65536 \
    --gen-tokens 512 --gen-warmup-tokens 32
```

At a limit, the guard sends `SIGTERM` to the process group, waits 10 seconds,
then uses `SIGKILL` if needed. A guarded run returns exit status 75 and records
the triggering metric in JSON. Set thresholds explicitly for diagnostic runs;
do not disable the host-availability floor on unified-memory systems. Existing
artifacts are not overwritten unless `--overwrite` is passed explicitly. Run
the synthetic guard tests with `make test-memory-guard`; they do not open a
model.

For CUDA allocation attribution, set `DS4_CUDA_MEMORY_REPORT=1`. The report
keeps logical model bytes separate from arena capacity and prints Q8 caches,
runtime tensor live/peak bytes, device-cache slabs, scratch/streaming capacity,
pinned-host staging, and the delta from `cudaMemGetInfo`. The delta includes
driver/runtime allocations and any CUDA allocation category not yet tracked;
it must not be added to the external watchdog peak as if it were separate
memory. Keep the external watchdog as the authoritative campaign gate.

### GB10 model-residency profiles

The GB10 memory campaign uses exact-size model arenas to avoid multi-GiB slack:

```sh
export DS4_CUDA_WEIGHT_ARENA_EXACT=1
```

For the 80.76 GiB Q2 tensor set, keep both the model cache and Q8-to-F16
cache unlimited. The complete 10.58 GiB Q8 cache is validated through 100K.

For the 90.88 GiB Q2-Q4 tensor set, use a context-aware weight residency while
keeping the Q8-to-F16 cache at 6144 MiB. Ranges beyond the model-cache limit
remain in the mmap and are accessed through HMM/UVA:

```sh
# Canonical through 65K.
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=90
export DS4_CUDA_Q8_F16_CACHE_MB=6144

# Use 89 GiB instead at 100K.
```

The 90 GiB profile completed 100K at 109.991 GB decimal, only 8.6 MB below
the watchdog, so that result is a demonstrated limit rather than an operational
profile. Use 89 GiB for 100K and keep the host otherwise idle.

For decode-heavy Q2-Q4 runs with a constrained model cache, the optional
placement below loads the output matrix before the normal ascending spans:

```sh
export DS4_CUDA_WEIGHT_CACHE_OUTPUT_FIRST=1
```

It does not increase the cache budget. On GB10 at 90/6 it improved median 2K
decode from 15.92 to 16.19 t/s, but reduced 65K prefill from 253.02 to 240.56
t/s while decode rose from 12.55 to 12.64 t/s. Keep it disabled for canonical
long-prefill runs; it is an explicit decode-heavy trade-off, not a default.

The phase-aware alternative preserves the normal ascending cache
during prefill and copies `output.weight` to a separate device allocation on
the first subsequent decode evaluation:

```sh
export DS4_CUDA_WEIGHT_CACHE_OUTPUT_PHASE_AWARE=1
```

It is CUDA single-device only, traditional decode only, and disabled by
default. For the current Q2-Q4 file it adds about 536.56 MiB after prefill; use
the memory guard and a 115 GB decimal ceiling. Model-backed quality passed. At
65K with 90/6, measured decode changed from 12.69 to 12.86 t/s and peak NVIDIA
memory was 109.803 GB. At 100K with 89/6, it changed from 12.00 to 12.15 t/s and
peak memory was 109.509 GB. The one-time 235-243 ms copy breaks even after
roughly 230-240 generated tokens, so enable it only for responses expected to
be longer than that. Do not combine it with
`DS4_CUDA_WEIGHT_CACHE_OUTPUT_FIRST`.

With the Q2 file and its complete model cache, `output.weight` is already
resident. The phase-aware flag was verified as a no-op: control and candidate
had the same 101,089,017,856-byte NVIDIA peak and decode stayed within noise.
Leave the flag disabled for Q2 because it provides no benefit.

The validated GB10 operating matrix is:

| Model | Context | Model cache | Q8-to-F16 | Phase-aware | Observed peak | Measured decode |
|---|---:|---:|---:|---|---:|---:|
| Q2 imatrix | 100K | unlimited | unlimited | off | 105.461 GB | use the normal Q2 profile |
| Q2-Q4 imatrix | 65K | 90 GiB | 6144 MiB | on for >~230 tokens | 109.803 GB | 12.86 t/s |
| Q2-Q4 imatrix | 100K | 89 GiB | 6144 MiB | on for >~240 tokens | 109.509 GB | 12.15 t/s |

Build these profiles with `make cuda-spark-gb10`. Always keep
`DS4_CUDA_WEIGHT_ARENA_EXACT=1`. If phase promotion fails, DS4 keeps the
original direct mapping and continues without the optimization. For a manual
fallback, unset the phase-aware flag; if more memory headroom is needed, reduce
the model-cache limit before lowering Q2-Q4 below its accepted 65K context
floor. These profiles do not reach the campaign's 20 t/s target.

`DS4_CUDA_DIRECT_MODEL=1` is diagnostic only on GB10: it
minimizes resident device memory but severely reduces prefill throughput.
