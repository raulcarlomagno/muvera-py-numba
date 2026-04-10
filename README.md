# muvera-numba

`muvera-numba` is a high-performance adaptation of the Fixed Dimensional Encoding (FDE) generator used in the [muvera-py](https://github.com/sionic-ai/muvera-py) project.

This project is intentionally lightweight: it is not meant to be installed as a mandatory dependency in every codebase.
If your project already uses `numpy` and `numba`, you can simply copy `fde_generator_numba.py` into your own repository and use it directly without adding extra packaging overhead.

The current version keeps the random generation path for SimHash, AMS sketch, and Count Sketch aligned with the original Python implementation so the outputs remain equivalent across both versions.
This is useful if you want to use `fde_generator_numba.py` for large batch generation while still keeping the original pure-Python implementation for query-time paths.
If your only goal is maximum Numba-oriented performance, that part could be re-optimized more aggressively, but doing so would break strict output equivalence with `muvera-py`.
That alternative direction is described in point 4 of [`CHANGES.md`](CHANGES.md): replacing Python-managed RNG usage with Numba-native random initialization to reduce overhead in repeated SimHash, AMS sketch, and Count Sketch setup.

The goal of this repository is simple:

- keep the original FDE behavior and API spirit
- accelerate the implementation with [Numba](https://numba.pydata.org/)
- add a batch-oriented query interface for more efficient multi-query processing

To avoid drifting away from the original project, this fork still follows the Python baseline used by `muvera-py` and targets `requires-python = ">=3.11"`.

## What this project does

This repository focuses on generating Fixed Dimensional Encodings for:

- query point clouds
- document point clouds
- batches of queries
- batches of documents

The Numba version reduces Python overhead by moving the heavy numerical work into JIT-compiled functions. That makes the encoding pipeline significantly faster, especially when processing many vectors or large batches.

## Main changes from the original implementation

The core source file, `fde_generator.py`, was taken from:

https://github.com/sionic-ai/muvera-py

It was then adapted into `fde_generator_numba.py` with the following changes:

- Numba JIT compilation for the heavy numerical paths, replacing pure Python loops with compiled native code
- parallel execution with `parallel=True` and `prange` in the main hotspots
- `fastmath=True` in the computationally dense kernels to let LLVM generate faster vectorized instructions
- in-place style aggregation logic to avoid expensive temporary allocations and reduce memory pressure
- contiguous array handling with `np.ascontiguousarray()` to improve cache locality during matrix operations
- a new `generate_query_fde_batch()` method that was not present in the original version and enables efficient batch query generation

For a more technical breakdown of the optimization work, see [`CHANGES.md`](CHANGES.md).

## Performance focus

The Numba version was designed to improve throughput in the parts of the pipeline that matter most:

- repeated FDE generation over many queries
- document aggregation across partitions
- projection-heavy matrix operations
- workloads where Python overhead becomes a bottleneck

In practice, this makes the project a better fit for high-volume embedding pipelines and retrieval systems that need to generate encodings quickly.

## Key features

- `generate_query_fde()`
- `generate_document_fde()`
- `generate_fde()`
- `generate_query_fde_batch()`
- `generate_document_fde_batch()`

## Configuration

The main configuration object is `FixedDimensionalEncodingConfig`.

Important options include:

- `dimension`: input embedding dimension
- `num_repetitions`: number of FDE repetitions
- `num_simhash_projections`: number of SimHash projections
- `projection_type`: identity projection or AMS sketch projection
- `projection_dimension`: output dimension for non-identity projections
- `fill_empty_partitions`: optional fill strategy for document encoding
- `final_projection_dimension`: optional Count Sketch projection at the end

## Example usage

```python
import numpy as np

from fde_generator_numba import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_query_fde_batch,
)

config = FixedDimensionalEncodingConfig(
    dimension=128,
    num_repetitions=2,
    num_simhash_projections=4,
    seed=42,
)

query = np.random.randn(32, 128).astype(np.float32)
fde = generate_query_fde(query, config)

batch = [
    np.random.randn(16, 128).astype(np.float32),
    np.random.randn(24, 128).astype(np.float32),
]
batch_fde = generate_query_fde_batch(batch, config)
```

## Notes

- The first call may take longer because Numba compiles the functions on demand.
- `generate_query_fde_batch()` is intended for batch query workloads and avoids calling the single-query function repeatedly.
- The codebase is still rooted in the Python implementation from `muvera-py`; the Numba version is an optimization layer, not a rewrite of the project model.

## Benchmarking

This repository keeps correctness checks and performance checks separate.

- Use `uv run pytest` for equivalence and regression testing.
- Use the benchmark script for timing comparisons between the original implementation and the Numba version.

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

`numba_cold_ms` is shown separately so the one-time JIT compilation cost is visible without distorting the steady-state throughput comparison.
Each benchmark case runs in a separate subprocess so JIT state is isolated and memory is released when that worker exits.

## Repository files

- `fde_generator.py`: original reference implementation
- `fde_generator_numba.py`: optimized Numba version
- `CHANGES.md`: summary of the main optimizations introduced in this fork
- `benchmarks/benchmark_fde.py`: reproducible speed comparison script

## License

This project is based on muvera-py, which states it follows the Apache 2.0 license, although no LICENSE file is included in the original repository.

This project assumes the same licensing intent.