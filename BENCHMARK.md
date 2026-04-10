# Benchmarking

This repository keeps correctness checks and performance checks separate.

- Use `uv run pytest` for equivalence and regression testing.
- Use the benchmark script for timing comparisons between the original
  implementation and the Numba version.

Run the benchmark with:

```bash
uv run python benchmarks/benchmark_fde.py
```

Useful options:

- `--repeats 3`: increase the number of timing repetitions
- `--cases query_single_large document_batch_large`: run only a subset of cases
- `--skip-validate`: skip the numerical equivalence check before timing

The benchmark reports:

- `original_ms`: median runtime of the baseline Python implementation
- `numba_cold_ms`: first Numba call in a fresh worker process
- `numba_warm_ms`: median runtime after an explicit warmup call
- `speedup`: `original_ms / numba_warm_ms`

`numba_cold_ms` is shown separately so the one-time JIT compilation cost is
visible without distorting the steady-state throughput comparison.
Each benchmark case runs in a separate subprocess so JIT state is isolated and
memory is released when that worker exits.

Example observations from a representative run with
`uv run python benchmarks/benchmark_fde.py --repeats 3`:

- Environment used for that sample run: `AMD Ryzen 7 7735HS`, `32 GB RAM`,
  `Windows 11` (system-reported build `26200`).
- Warm Numba throughput was consistently faster than the baseline, with
  observed speedups ranging from `4.64x` to `26.14x`.
- Cold Numba timings were dominated by JIT compilation, so they are much less
  relevant for steady-state services or repeated batch jobs.
- The strongest gains appeared in document workloads and in cases using
  `final_projection_dimension`, where the original Python implementation pays
  more overhead for the extra Count Sketch pass.
- Plain query paths still improved clearly, but less dramatically than document
  paths because their baseline logic is simpler and already relatively cheap at
  small sizes.

| Workload family | Representative cases | Observed warm speedup |
| --- | --- | --- |
| Query single | `query_single_small` to `query_single_large` | `4.66x` to `6.08x` |
| Query single + final projection | `query_single_final_projection` | `15.36x` |
| Document single | `document_single_small` to `document_single_large` | `5.87x` to `26.14x` |
| Document single + final projection | `document_single_final_projection` | `17.62x` |
| Query batch | `query_batch_small` to `query_batch_large` | `4.64x` to `5.15x` |
| Query batch + final projection | `query_batch_final_projection` | `14.63x` |
| Document batch | `document_batch_small` to `document_batch_large` | `5.08x` to `8.33x` |
| Document batch + final projection | `document_batch_final_projection` | `18.49x` |

Treat those numbers as machine-specific examples rather than fixed guarantees,
but they are a good summary of the current performance profile: warm Numba
execution helps across the board, and the relative advantage becomes larger as
aggregation work and final projection work increase.
