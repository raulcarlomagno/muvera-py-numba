# muvera-py-numba

Numba-accelerated MUVERA Fixed Dimensional Encoding (FDE) generation based on
the original implementation from
[`muvera-py`](https://github.com/sionic-ai/muvera-py).

In the current benchmark suite, warm execution showed observed speedups between
`4x` and `26x` versus the baseline Python version.

This repository aims to stay close to the original API and behavior while
improving throughput for repeated and batch-oriented workloads.

It is intentionally lightweight: it is not meant to be a mandatory dependency
in every codebase. If your project already uses `numpy` and `numba`, you can
simply copy `fde_generator_numba.py` into your own repository and use it
directly without adding extra packaging overhead.

To stay close to the original project, this fork follows the Python baseline
used by `muvera-py` and targets Python `>=3.11`.

## Highlights

- keeps behavior aligned with the original `muvera-py` implementation
- accelerates the heavy numerical paths with [Numba](https://numba.pydata.org/)
- adds `generate_query_fde_batch()` for more efficient multi-query processing
- can be vendored easily if your project already uses `numpy` and `numba`

## Main API

- `generate_query_fde()`
- `generate_document_fde()`
- `generate_fde()`
- `generate_query_fde_batch()`
- `generate_document_fde_batch()`

The main configuration object is `FixedDimensionalEncodingConfig`.

## Example

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

- The first call may be slower because Numba compiles functions on demand.
- The current version preserves the original random-generation path so outputs
  remain comparable with `muvera-py`.

## Benchmarking

Use `uv run pytest` for equivalence and regression checks.

Run the benchmark suite with:

```bash
uv run python benchmarks/benchmark_fde.py
```

In the current sample runs, warm execution was consistently faster than the
baseline, with the largest gains appearing in document-heavy workloads and in
configurations that use `final_projection_dimension`. Cold timings are much less
representative because they include one-time JIT compilation overhead.

For the full benchmark methodology and sample results, see
[`BENCHMARK.md`](BENCHMARK.md).

For optimization details, see [`CHANGES.md`](CHANGES.md).

## License

This project is based on `muvera-py`, which states Apache 2.0 licensing intent.
No `LICENSE` file is included in the upstream repository, so this repository
assumes the same intent.
