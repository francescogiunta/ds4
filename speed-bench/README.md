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
