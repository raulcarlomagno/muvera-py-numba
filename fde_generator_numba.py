import logging
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional

import numpy as np
from numba import njit, prange


class EncodingType(Enum):
    DEFAULT_SUM = 0
    AVERAGE = 1


class ProjectionType(Enum):
    DEFAULT_IDENTITY = 0
    AMS_SKETCH = 1


@dataclass
class FixedDimensionalEncodingConfig:
    dimension: int = 128
    num_repetitions: int = 10
    num_simhash_projections: int = 6
    seed: int = 42
    encoding_type: EncodingType = EncodingType.DEFAULT_SUM
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: Optional[int] = None
    fill_empty_partitions: bool = False
    final_projection_dimension: Optional[int] = None


_PROFILE_ENABLED = os.environ.get("MUVERA_NUMBA_PROFILE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_SIMHASH_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_AMS_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_COUNT_SKETCH_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _profile_log(enabled: bool, label: str, start_time: float) -> None:
    if enabled:
        logging.info("[FDE Numba] %s: %.4fs", label, time.perf_counter() - start_time)


def _projection_dimension(config: FixedDimensionalEncodingConfig) -> int:
    if config.projection_type == ProjectionType.DEFAULT_IDENTITY:
        return config.dimension
    if not config.projection_dimension or config.projection_dimension <= 0:
        raise ValueError(
            "A positive projection_dimension is required for non-identity projections."
        )
    return config.projection_dimension


@njit(fastmath=True)
def _gray_code_to_binary_numba(num: int) -> int:
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num


@njit(parallel=True, fastmath=True)
def _compute_partition_indices_numba(
    sketches: np.ndarray, num_projections: int
) -> np.ndarray:
    n = sketches.shape[0]
    partition_indices = np.zeros(n, dtype=np.uint32)
    for i in prange(n):
        idx = 0
        for j in range(num_projections):
            bit = sketches[i, j] > 0
            idx = (idx << 1) + (int(bit) ^ (idx & 1))
        partition_indices[i] = idx
    return partition_indices


@njit(fastmath=True)
def _aggregate_sum_single_numba(
    projected_points: np.ndarray, partition_indices: np.ndarray, num_partitions: int
) -> np.ndarray:
    projection_dim = projected_points.shape[1]
    out = np.zeros(num_partitions * projection_dim, dtype=np.float32)
    for i in range(projected_points.shape[0]):
        part_idx = partition_indices[i]
        base = part_idx * projection_dim
        for d in range(projection_dim):
            out[base + d] += projected_points[i, d]
    return out


@njit(fastmath=True)
def _aggregate_avg_single_numba(
    projected_points: np.ndarray, partition_indices: np.ndarray, num_partitions: int
) -> tuple[np.ndarray, np.ndarray]:
    projection_dim = projected_points.shape[1]
    out = np.zeros((num_partitions, projection_dim), dtype=np.float32)
    counts = np.zeros(num_partitions, dtype=np.int32)

    for i in range(projected_points.shape[0]):
        part_idx = partition_indices[i]
        counts[part_idx] += 1
        for d in range(projection_dim):
            out[part_idx, d] += projected_points[i, d]

    for p in range(num_partitions):
        count = counts[p]
        if count > 0:
            for d in range(projection_dim):
                out[p, d] /= count

    return out, counts


@njit(fastmath=True)
def _fill_empty_partitions_single_numba(
    out: np.ndarray,
    counts: np.ndarray,
    sketches: np.ndarray,
    projected_points: np.ndarray,
    num_partitions: int,
    num_simhash_projections: int,
) -> None:
    for p in range(num_partitions):
        if counts[p] != 0:
            continue

        binary_rep = _gray_code_to_binary_numba(p)
        min_dist = num_simhash_projections + 1
        nearest_idx = 0

        for i in range(sketches.shape[0]):
            dist = 0
            for b in range(num_simhash_projections):
                sketch_bit = sketches[i, b] > 0
                target_bit = (binary_rep >> (num_simhash_projections - 1 - b)) & 1
                if sketch_bit != target_bit:
                    dist += 1

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        for d in range(projected_points.shape[1]):
            out[p, d] = projected_points[nearest_idx, d]


@njit(parallel=True, fastmath=True)
def _aggregate_sum_batch_numba_parallel(
    projected_points: np.ndarray,
    doc_boundaries: np.ndarray,
    partition_indices: np.ndarray,
    num_docs: int,
    num_partitions: int,
    projection_dim: int,
) -> np.ndarray:
    out = np.zeros((num_docs, num_partitions * projection_dim), dtype=np.float32)
    for doc_idx in prange(num_docs):
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx + 1]
        out_row = out[doc_idx]
        for i in range(start, end):
            base = partition_indices[i] * projection_dim
            for d in range(projection_dim):
                out_row[base + d] += projected_points[i, d]
    return out


@njit(parallel=True, fastmath=True)
def _aggregate_avg_batch_numba_parallel(
    projected_points: np.ndarray,
    doc_boundaries: np.ndarray,
    partition_indices: np.ndarray,
    num_docs: int,
    num_partitions: int,
    projection_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((num_docs, num_partitions, projection_dim), dtype=np.float32)
    counts = np.zeros((num_docs, num_partitions), dtype=np.int32)

    for doc_idx in prange(num_docs):
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx + 1]
        out_doc = out[doc_idx]
        counts_doc = counts[doc_idx]
        for i in range(start, end):
            part_idx = partition_indices[i]
            counts_doc[part_idx] += 1
            out_part = out_doc[part_idx]
            for d in range(projection_dim):
                out_part[d] += projected_points[i, d]

        for p in range(num_partitions):
            count = counts_doc[p]
            if count > 0:
                out_part = out_doc[p]
                for d in range(projection_dim):
                    out_part[d] /= count

    return out, counts


@njit(parallel=True, fastmath=True)
def _fill_empty_partitions_numba(
    out: np.ndarray,
    counts: np.ndarray,
    doc_boundaries: np.ndarray,
    doc_sketches: np.ndarray,
    projected_points: np.ndarray,
    num_docs: int,
    num_partitions: int,
    projection_dim: int,
    num_simhash_projections: int,
) -> None:
    for doc_idx in prange(num_docs):
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx + 1]
        if start == end:
            continue

        for p in range(num_partitions):
            if counts[doc_idx, p] != 0:
                continue

            binary_rep = _gray_code_to_binary_numba(p)
            min_dist = num_simhash_projections + 1
            nearest_idx = start

            for i in range(start, end):
                dist = 0
                for b in range(num_simhash_projections):
                    sketch_bit = doc_sketches[i, b] > 0
                    target_bit = (binary_rep >> (num_simhash_projections - 1 - b)) & 1
                    if sketch_bit != target_bit:
                        dist += 1

                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            for d in range(projection_dim):
                out[doc_idx, p, d] = projected_points[nearest_idx, d]


@njit(fastmath=True)
def _apply_count_sketch_vector_numba(
    input_vector: np.ndarray, indices: np.ndarray, signs: np.ndarray, final_dimension: int
) -> np.ndarray:
    out = np.zeros(final_dimension, dtype=np.float32)
    for i in range(input_vector.shape[0]):
        out[indices[i]] += signs[i] * input_vector[i]
    return out


@njit(parallel=True, fastmath=True)
def _apply_count_sketch_batch_numba(
    input_matrix: np.ndarray, indices: np.ndarray, signs: np.ndarray, final_dimension: int
) -> np.ndarray:
    num_docs = input_matrix.shape[0]
    out = np.zeros((num_docs, final_dimension), dtype=np.float32)
    for i in prange(num_docs):
        for j in range(input_matrix.shape[1]):
            out[i, indices[j]] += signs[j] * input_matrix[i, j]
    return out


def _simhash_matrix_from_seed(
    dimension: int, num_projections: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections)).astype(
        np.float32
    )


def _ams_projection_matrix_from_seed(
    dimension: int, projection_dim: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((dimension, projection_dim), dtype=np.float32)
    indices = rng.integers(0, projection_dim, size=dimension)
    signs = rng.choice([-1.0, 1.0], size=dimension)
    out[np.arange(dimension), indices] = signs
    return out


def _generate_count_sketch_params(seed: int, size: int, final_dimension: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, final_dimension, size=size).astype(np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=size).astype(
        np.float32
    )
    return indices, signs


def _get_simhash_matrix(dimension: int, num_projections: int, seed: int) -> np.ndarray:
    key = (dimension, num_projections, seed)
    matrix = _SIMHASH_CACHE.get(key)
    if matrix is None:
        matrix = _simhash_matrix_from_seed(dimension, num_projections, seed)
        _SIMHASH_CACHE[key] = matrix
    return matrix


def _get_ams_projection_matrix(
    dimension: int, projection_dim: int, seed: int
) -> np.ndarray:
    key = (dimension, projection_dim, seed)
    matrix = _AMS_CACHE.get(key)
    if matrix is None:
        matrix = _ams_projection_matrix_from_seed(dimension, projection_dim, seed)
        _AMS_CACHE[key] = matrix
    return matrix


def _get_count_sketch_params(
    seed: int, size: int, final_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    key = (seed, size, final_dimension)
    params = _COUNT_SKETCH_CACHE.get(key)
    if params is None:
        params = _generate_count_sketch_params(seed, size, final_dimension)
        _COUNT_SKETCH_CACHE[key] = params
    return params


def _project_points(
    point_cloud: np.ndarray,
    config: FixedDimensionalEncodingConfig,
    current_seed: int,
    projection_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    simhash_matrix = _get_simhash_matrix(
        config.dimension, config.num_simhash_projections, current_seed
    )
    sketches = point_cloud @ simhash_matrix

    if config.projection_type == ProjectionType.DEFAULT_IDENTITY:
        return sketches, point_cloud

    ams_matrix = _get_ams_projection_matrix(config.dimension, projection_dim, current_seed)
    return sketches, point_cloud @ ams_matrix


def _generate_query_fde_single(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    if point_cloud.ndim != 2 or point_cloud.shape[1] != config.dimension:
        raise ValueError(
            f"Input data shape {point_cloud.shape} is inconsistent with config dimension {config.dimension}."
        )
    if not (0 <= config.num_simhash_projections < 32):
        raise ValueError(
            f"num_simhash_projections must be in [0, 31]: {config.num_simhash_projections}"
        )

    point_cloud = np.ascontiguousarray(point_cloud.astype(np.float32))
    projection_dim = _projection_dimension(config)
    num_partitions = 2**config.num_simhash_projections
    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fde = np.zeros(final_fde_dim, dtype=np.float32)
    profile = _PROFILE_ENABLED

    for rep_num in range(config.num_repetitions):
        rep_seed = config.seed + rep_num
        if profile:
            start = time.perf_counter()
            sketches, projected_points = _project_points(
                point_cloud, config, rep_seed, projection_dim
            )
            _profile_log(profile, f"single_query rep={rep_num} projections", start)

            start = time.perf_counter()
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out = _aggregate_sum_single_numba(
                projected_points, partition_indices, num_partitions
            )
            _profile_log(profile, f"single_query rep={rep_num} aggregate", start)
        else:
            sketches, projected_points = _project_points(
                point_cloud, config, rep_seed, projection_dim
            )
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out = _aggregate_sum_single_numba(
                projected_points, partition_indices, num_partitions
            )

        rep_start = rep_num * num_partitions * projection_dim
        out_fde[rep_start : rep_start + rep_fde_out.size] = rep_fde_out

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        indices, signs = _get_count_sketch_params(
            config.seed, final_fde_dim, config.final_projection_dimension
        )
        out_fde = _apply_count_sketch_vector_numba(
            out_fde, indices, signs, config.final_projection_dimension
        )

    return out_fde


def _generate_document_fde_single(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    if point_cloud.ndim != 2 or point_cloud.shape[1] != config.dimension:
        raise ValueError(
            f"Input data shape {point_cloud.shape} is inconsistent with config dimension {config.dimension}."
        )
    if not (0 <= config.num_simhash_projections < 32):
        raise ValueError(
            f"num_simhash_projections must be in [0, 31]: {config.num_simhash_projections}"
        )

    point_cloud = np.ascontiguousarray(point_cloud.astype(np.float32))
    projection_dim = _projection_dimension(config)
    num_partitions = 2**config.num_simhash_projections
    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fde = np.zeros(final_fde_dim, dtype=np.float32)
    profile = _PROFILE_ENABLED

    for rep_num in range(config.num_repetitions):
        rep_seed = config.seed + rep_num
        if profile:
            start = time.perf_counter()
            sketches, projected_points = _project_points(
                point_cloud, config, rep_seed, projection_dim
            )
            _profile_log(profile, f"single_document rep={rep_num} projections", start)

            start = time.perf_counter()
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out, counts = _aggregate_avg_single_numba(
                projected_points, partition_indices, num_partitions
            )
            if config.fill_empty_partitions and point_cloud.shape[0] > 0:
                _fill_empty_partitions_single_numba(
                    rep_fde_out,
                    counts,
                    sketches,
                    projected_points,
                    num_partitions,
                    config.num_simhash_projections,
                )
            _profile_log(profile, f"single_document rep={rep_num} aggregate", start)
        else:
            sketches, projected_points = _project_points(
                point_cloud, config, rep_seed, projection_dim
            )
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out, counts = _aggregate_avg_single_numba(
                projected_points, partition_indices, num_partitions
            )
            if config.fill_empty_partitions and point_cloud.shape[0] > 0:
                _fill_empty_partitions_single_numba(
                    rep_fde_out,
                    counts,
                    sketches,
                    projected_points,
                    num_partitions,
                    config.num_simhash_projections,
                )

        rep_start = rep_num * num_partitions * projection_dim
        out_fde[rep_start : rep_start + num_partitions * projection_dim] = rep_fde_out.reshape(
            -1
        )

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        indices, signs = _get_count_sketch_params(
            config.seed, final_fde_dim, config.final_projection_dimension
        )
        out_fde = _apply_count_sketch_vector_numba(
            out_fde, indices, signs, config.final_projection_dimension
        )

    return out_fde


def generate_document_fde_batch(
    doc_embeddings_list: List[np.ndarray], config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    num_docs = len(doc_embeddings_list)
    if num_docs == 0:
        return np.array([])

    valid_docs = []
    for i, doc in enumerate(doc_embeddings_list):
        if doc.ndim != 2:
            continue
        if doc.shape[1] != config.dimension:
            raise ValueError(
                f"Document {i} has incorrect dimension: expected {config.dimension}, got {doc.shape[1]}"
            )
        if doc.shape[0] == 0:
            continue
        valid_docs.append(doc)

    if not valid_docs:
        return np.array([])

    projection_dim = _projection_dimension(config)
    num_partitions = 2**config.num_simhash_projections
    doc_lengths = np.array([len(doc) for doc in valid_docs], dtype=np.int32)
    doc_boundaries = np.insert(np.cumsum(doc_lengths), 0, 0)
    all_points = np.ascontiguousarray(np.vstack(valid_docs).astype(np.float32))
    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fdes = np.zeros((len(valid_docs), final_fde_dim), dtype=np.float32)
    profile = _PROFILE_ENABLED

    for rep_num in range(config.num_repetitions):
        rep_seed = config.seed + rep_num
        if profile:
            start = time.perf_counter()
            sketches, projected_points = _project_points(
                all_points, config, rep_seed, projection_dim
            )
            _profile_log(profile, f"batch_document rep={rep_num} projections", start)

            start = time.perf_counter()
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            _profile_log(profile, f"batch_document rep={rep_num} partition_indices", start)

            start = time.perf_counter()
            rep_fde_out, partition_counts = _aggregate_avg_batch_numba_parallel(
                projected_points,
                doc_boundaries,
                partition_indices,
                len(valid_docs),
                num_partitions,
                projection_dim,
            )
            _profile_log(profile, f"batch_document rep={rep_num} aggregate", start)

            if config.fill_empty_partitions:
                start = time.perf_counter()
                _fill_empty_partitions_numba(
                    rep_fde_out,
                    partition_counts,
                    doc_boundaries,
                    sketches,
                    projected_points,
                    len(valid_docs),
                    num_partitions,
                    projection_dim,
                    config.num_simhash_projections,
                )
                _profile_log(profile, f"batch_document rep={rep_num} fill_empty", start)
        else:
            sketches, projected_points = _project_points(
                all_points, config, rep_seed, projection_dim
            )
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out, partition_counts = _aggregate_avg_batch_numba_parallel(
                projected_points,
                doc_boundaries,
                partition_indices,
                len(valid_docs),
                num_partitions,
                projection_dim,
            )
            if config.fill_empty_partitions:
                _fill_empty_partitions_numba(
                    rep_fde_out,
                    partition_counts,
                    doc_boundaries,
                    sketches,
                    projected_points,
                    len(valid_docs),
                    num_partitions,
                    projection_dim,
                    config.num_simhash_projections,
                )

        rep_start = rep_num * num_partitions * projection_dim
        out_fdes[:, rep_start : rep_start + num_partitions * projection_dim] = rep_fde_out.reshape(
            len(valid_docs), -1
        )

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        if profile:
            start = time.perf_counter()
            indices, signs = _get_count_sketch_params(
                config.seed, final_fde_dim, config.final_projection_dimension
            )
            out_fdes = _apply_count_sketch_batch_numba(
                out_fdes, indices, signs, config.final_projection_dimension
            )
            _profile_log(profile, "batch_document final_projection", start)
        else:
            indices, signs = _get_count_sketch_params(
                config.seed, final_fde_dim, config.final_projection_dimension
            )
            out_fdes = _apply_count_sketch_batch_numba(
                out_fdes, indices, signs, config.final_projection_dimension
            )

    return out_fdes


def generate_query_fde_batch(
    query_embeddings_list: list[np.ndarray], config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    num_queries = len(query_embeddings_list)
    if num_queries == 0:
        return np.array([])
    if config.fill_empty_partitions:
        raise ValueError("Query FDE batch does not support 'fill_empty_partitions'.")

    valid_queries = []
    for i, query in enumerate(query_embeddings_list):
        if query.ndim != 2:
            continue
        if query.shape[1] != config.dimension:
            raise ValueError(
                f"Query {i} has incorrect dimension: expected {config.dimension}, got {query.shape[1]}"
            )
        if query.shape[0] == 0:
            continue
        valid_queries.append(query)

    if not valid_queries:
        return np.array([])

    projection_dim = _projection_dimension(config)
    num_partitions = 2**config.num_simhash_projections
    query_lengths = np.array([len(query) for query in valid_queries], dtype=np.int32)
    query_boundaries = np.insert(np.cumsum(query_lengths), 0, 0)
    all_points = np.ascontiguousarray(np.vstack(valid_queries).astype(np.float32))
    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fdes = np.zeros((len(valid_queries), final_fde_dim), dtype=np.float32)
    profile = _PROFILE_ENABLED

    for rep_num in range(config.num_repetitions):
        rep_seed = config.seed + rep_num
        if profile:
            start = time.perf_counter()
            sketches, projected_points = _project_points(
                all_points, config, rep_seed, projection_dim
            )
            _profile_log(profile, f"batch_query rep={rep_num} projections", start)

            start = time.perf_counter()
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out = _aggregate_sum_batch_numba_parallel(
                projected_points,
                query_boundaries,
                partition_indices,
                len(valid_queries),
                num_partitions,
                projection_dim,
            )
            _profile_log(profile, f"batch_query rep={rep_num} aggregate", start)
        else:
            sketches, projected_points = _project_points(
                all_points, config, rep_seed, projection_dim
            )
            partition_indices = _compute_partition_indices_numba(
                sketches, config.num_simhash_projections
            )
            rep_fde_out = _aggregate_sum_batch_numba_parallel(
                projected_points,
                query_boundaries,
                partition_indices,
                len(valid_queries),
                num_partitions,
                projection_dim,
            )

        rep_start = rep_num * num_partitions * projection_dim
        out_fdes[:, rep_start : rep_start + num_partitions * projection_dim] = rep_fde_out

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        if profile:
            start = time.perf_counter()
            indices, signs = _get_count_sketch_params(
                config.seed, final_fde_dim, config.final_projection_dimension
            )
            out_fdes = _apply_count_sketch_batch_numba(
                out_fdes, indices, signs, config.final_projection_dimension
            )
            _profile_log(profile, "batch_query final_projection", start)
        else:
            indices, signs = _get_count_sketch_params(
                config.seed, final_fde_dim, config.final_projection_dimension
            )
            out_fdes = _apply_count_sketch_batch_numba(
                out_fdes, indices, signs, config.final_projection_dimension
            )

    return out_fdes


def _generate_fde_internal(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return _generate_query_fde_single(point_cloud, config)
    return _generate_document_fde_single(point_cloud, config)


def generate_query_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """Generates a Fixed Dimensional Encoding for a query point cloud (using SUM)."""
    if config.fill_empty_partitions:
        raise ValueError(
            "Query FDE generation does not support 'fill_empty_partitions'."
        )
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return _generate_fde_internal(point_cloud, query_config)


def generate_document_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """Generates a Fixed Dimensional Encoding for a document point cloud (using AVERAGE)."""
    doc_config = replace(config, encoding_type=EncodingType.AVERAGE)
    return _generate_fde_internal(point_cloud, doc_config)


def generate_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return generate_query_fde(point_cloud, config)
    if config.encoding_type == EncodingType.AVERAGE:
        return generate_document_fde(point_cloud, config)
    raise ValueError(f"Unsupported encoding type in config: {config.encoding_type}")


if __name__ == "__main__":
    print(f"\n{'=' * 20} SCENARIO 1: Basic FDE Generation (NUMBA) {'=' * 20}")
    base_config = FixedDimensionalEncodingConfig(
        dimension=128, num_repetitions=2, num_simhash_projections=4, seed=42
    )
    query_data = np.random.randn(32, base_config.dimension).astype(np.float32)
    doc_data = np.random.randn(80, base_config.dimension).astype(np.float32)

    t0 = time.perf_counter()
    query_fde = generate_query_fde(query_data, base_config)
    t1 = time.perf_counter()
    print(f"Query FDE gen time (incl compilation): {t1-t0:.4f}s")

    doc_fde = generate_document_fde(
        doc_data, replace(base_config, fill_empty_partitions=True)
    )

    expected_dim = (
        base_config.num_repetitions
        * (2**base_config.num_simhash_projections)
        * base_config.dimension
    )
    print(f"Query FDE Shape: {query_fde.shape} (Expected: {expected_dim})")
    print(f"Document FDE Shape: {doc_fde.shape} (Expected: {expected_dim})")
    print(f"Similarity Score: {np.dot(query_fde, doc_fde):.4f}")
    assert query_fde.shape[0] == expected_dim

    print(f"\n{'=' * 20} SCENARIO 2: Inner Projection (AMS Sketch) {'=' * 20}")
    ams_config = replace(
        base_config, projection_type=ProjectionType.AMS_SKETCH, projection_dimension=16
    )
    query_fde_ams = generate_query_fde(query_data, ams_config)
    expected_dim_ams = (
        ams_config.num_repetitions
        * (2**ams_config.num_simhash_projections)
        * ams_config.projection_dimension
    )
    print(f"AMS Sketch FDE Shape: {query_fde_ams.shape} (Expected: {expected_dim_ams})")
    assert query_fde_ams.shape[0] == expected_dim_ams

    print(f"\n{'=' * 20} SCENARIO 3: Final Projection (Count Sketch) {'=' * 20}")
    final_proj_config = replace(base_config, final_projection_dimension=1024)
    query_fde_final = generate_query_fde(query_data, final_proj_config)
    print(
        f"Final Projection FDE Shape: {query_fde_final.shape} (Expected: {final_proj_config.final_projection_dimension})"
    )
    assert query_fde_final.shape[0] == final_proj_config.final_projection_dimension

    print(f"\n{'=' * 20} SCENARIO 4: Batch Processing {'=' * 20}")
    query_data_list = [
        np.random.randn(np.random.randint(10, 50), base_config.dimension).astype(np.float32)
        for _ in range(5)
    ]
    query_fde_batch = generate_query_fde_batch(query_data_list, base_config)

    print(f"Batch Output shape: {query_fde_batch.shape}")
    assert query_fde_batch.shape == (5, expected_dim)

    print("\nAll test scenarios completed successfully.")
