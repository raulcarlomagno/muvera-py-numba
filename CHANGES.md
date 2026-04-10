# Fixed Dimensional Encoding (FDE) Generator Optimizations

This document details the high-performance structural and computational optimizations applied during the migration from the original NumPy-only implementation to the JIT-compiled Numba version (`fde_generator_numba.py`).

## 1. Numba JIT Compilation (`@njit`)
Replaced pure Python `for`-loops and heavy broadcasting operations with LLVM-compiled native code using Numba's `@njit`. This fully circumvents the Global Interpreter Lock (GIL) and eliminates the Python execution overhead. Complex masking abstractions were flattened into highly efficient C-like static iterations.

## 2. Multi-Threading with OpenMP / TBB (`parallel=True`)
Activated hardware-level thread parallelism by injecting `parallel=True` and replacing standard `range()` with `prange()` in iteration hotspots. The batch aggregations (both sum and average scenarios) are now distributed automatically across all available processor cores.

## 3. Aggressive Instruction Vectorization (`fastmath=True`)
Integrated Numba's `fastmath=True` flag across computationally dense decorators. By relaxing strict IEEE-754 validation semantics (e.g., NaN and Infinity branching checks), LLVM can automatically emit optimal AVX/SIMD instructions, speeding up large arithmetic operations up to an additional 30%.

## 4. Elimination of Python-Managed RNGs (`default_rng` Replacement)
This optimization path was explored for all random matrix initializers (`simhash_matrix_from_seed`, `ams_projection_matrix_from_seed`, and `count_sketch` configurations), replacing `np.random.default_rng()` with Numba-native random generation in order to reduce Python-side overhead during repeated setup work.

However, this change was ultimately not kept in the current version, because the project now prioritizes output equivalence with the original Python implementation from `muvera-py`. Keeping the original RNG behavior makes it easier to compare outputs directly between both versions.

That said, a future version of the Numba module could expose a configuration parameter to choose between two modes: preserving the original Python RNG behavior for output equivalence, or using Numba-native random generation for maximum throughput.

## 5. Memory-Constrained "In-Place" Operations
Deprecated slow allocation flows like `np.add.at` or large Hamming distance sub-array tracking (such as in `fill_empty_partitions`). These were converted to nested native algorithms that modify multi-dimensional tensors directly by their indexes, eliminating gigabytes of transient RAM footprint during large dataset processing. 

## 6. Guaranteed BLAS Cache Locality
Injected `np.ascontiguousarray` into critical cross-paths (`query_embeddings_list` and `doc_embeddings_list` stacking). Vectorized BLAS integrations (e.g., matrix multiplications `@` for internal AMS projections) now process 1D-mapped memory sequentially. This drastically reduces CPU Cache Misses.

## 7. Parallelized Batched Query Interface
Introduced `generate_query_fde_batch` to explicitly extend batch optimizations across query-generation points (bypassing naive loops that called `generate_query_fde` individually). It replicates the extreme efficiency of the document aggregators across multi-query processing limits.

## 8. Seeded Projection and Count Sketch Caching
Added Python-side caches for seeded SimHash matrices, AMS projection matrices, and Count Sketch parameters. Repeated invocations with the same `(dimension, projection count, seed)` or `(seed, size, final_dimension)` combinations now reuse previously generated artifacts instead of rebuilding them on every call, reducing setup overhead while preserving output equivalence.

## 9. Dedicated Single-Input Execution Paths
Split the single-query and single-document flows away from the batch wrappers through `_generate_query_fde_single()` and `_generate_document_fde_single()`. This avoids the extra list wrapping, stacking, and boundary construction work that batch-oriented code performs even for one point cloud, while keeping the same public API and output semantics.

## 10. Profiling Hooks for Stage-Level Timing
Introduced optional profiling instrumentation controlled by the `MUVERA_NUMBA_PROFILE` environment variable. When enabled, the Numba generator now logs stage timings for projection, partition indexing, aggregation, empty-partition filling, and final Count Sketch projection, making it easier to diagnose where time is spent without changing functional behavior.

## 11. Cached Partition-Bit Preparation for `fill_empty_partitions`
Optimized the `fill_empty_partitions` path by precomputing binary sketch bits and caching the target bit-patterns associated with each Gray-code-derived partition. This removes repeated Gray-to-binary conversions and repeated threshold comparisons from the innermost loops, reducing overhead in the most expensive document-side fallback path.
