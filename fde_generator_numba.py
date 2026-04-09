import logging
import time
import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List
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


@njit(fastmath=True)
def _gray_code_to_binary_numba(num: int) -> int:
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num


@njit(parallel=True, fastmath=True)
def _compute_partition_indices_numba(sketches: np.ndarray, num_projections: int) -> np.ndarray:
    n = sketches.shape[0]
    partition_indices = np.zeros(n, dtype=np.uint32)
    for i in prange(n):
        idx = 0
        for j in range(num_projections):
            bit = sketches[i, j] > 0
            idx = (idx << 1) + (int(bit) ^ (idx & 1))
        partition_indices[i] = idx
    return partition_indices


@njit(parallel=True, fastmath=True)
def _aggregate_sum_batch_numba_parallel(
    projected_points: np.ndarray,
    doc_boundaries: np.ndarray,
    partition_indices: np.ndarray,
    num_docs: int,
    num_partitions: int,
    projection_dim: int
) -> np.ndarray:
    out = np.zeros((num_docs, num_partitions * projection_dim), dtype=np.float32)
    for doc_idx in prange(num_docs):
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx + 1]
        for i in range(start, end):
            part_idx = partition_indices[i]
            for d in range(projection_dim):
                out[doc_idx, part_idx * projection_dim + d] += projected_points[i, d]
    return out


@njit(parallel=True, fastmath=True)
def _aggregate_avg_batch_numba_parallel(
    projected_points: np.ndarray,
    doc_boundaries: np.ndarray,
    partition_indices: np.ndarray,
    num_docs: int,
    num_partitions: int,
    projection_dim: int
):
    out = np.zeros((num_docs, num_partitions, projection_dim), dtype=np.float32)
    counts = np.zeros((num_docs, num_partitions), dtype=np.int32)
    
    for doc_idx in prange(num_docs):
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx+1]
        for i in range(start, end):
            part_idx = partition_indices[i]
            counts[doc_idx, part_idx] += 1
            for d in range(projection_dim):
                out[doc_idx, part_idx, d] += projected_points[i, d]
                
        # average
        for p in range(num_partitions):
            c = counts[doc_idx, p]
            if c > 0:
                for d in range(projection_dim):
                    out[doc_idx, p, d] /= c
                    
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
    num_simhash_projections: int
):
    for doc_idx in prange(num_docs):
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx+1]
        if start == end:
            continue
            
        for p in range(num_partitions):
            if counts[doc_idx, p] == 0:
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


@njit(parallel=True, fastmath=True)
def _apply_count_sketch_batch_numba(input_matrix: np.ndarray, indices: np.ndarray, signs: np.ndarray, final_dimension: int) -> np.ndarray:
    num_docs = input_matrix.shape[0]
    out = np.zeros((num_docs, final_dimension), dtype=np.float32)
    for i in prange(num_docs):
        for j in range(input_matrix.shape[1]):
            out[i, indices[j]] += signs[j] * input_matrix[i, j]
    return out


@njit(fastmath=True)
def _simhash_matrix_from_seed(
    dimension: int, num_projections: int, seed: int
) -> np.ndarray:
    np.random.seed(seed)
    out = np.empty((dimension, num_projections), dtype=np.float32)
    for i in range(dimension):
        for j in range(num_projections):
            out[i, j] = np.random.standard_normal()
    return out


@njit(fastmath=True)
def _ams_projection_matrix_from_seed(
    dimension: int, projection_dim: int, seed: int
) -> np.ndarray:
    np.random.seed(seed)
    out = np.zeros((dimension, projection_dim), dtype=np.float32)
    for i in range(dimension):
        idx = np.random.randint(0, projection_dim)
        sign = 1.0 if np.random.rand() > 0.5 else -1.0
        out[i, idx] = sign
    return out


@njit(fastmath=True)
def _generate_count_sketch_params_numba(seed: int, size: int, final_dimension: int):
    np.random.seed(seed)
    indices = np.empty(size, dtype=np.int32)
    signs = np.empty(size, dtype=np.float32)
    for i in range(size):
        indices[i] = np.random.randint(0, final_dimension)
        signs[i] = 1.0 if np.random.rand() > 0.5 else -1.0
    return indices, signs


def generate_document_fde_batch(
    doc_embeddings_list: List[np.ndarray], config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    batch_start_time = time.perf_counter()
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

    num_docs = len(valid_docs)
    doc_embeddings_list = valid_docs

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    if use_identity_proj:
        projection_dim = config.dimension
    else:
        if not config.projection_dimension or config.projection_dimension <= 0:
            raise ValueError(
                "A positive projection_dimension must be specified for non-identity projections"
            )
        projection_dim = config.projection_dimension

    num_partitions = 2**config.num_simhash_projections

    doc_lengths = np.array([len(doc) for doc in doc_embeddings_list], dtype=np.int32)
    doc_boundaries = np.insert(np.cumsum(doc_lengths), 0, 0)
    all_points = np.ascontiguousarray(np.vstack(doc_embeddings_list).astype(np.float32))

    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)

    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num

        simhash_matrix = _simhash_matrix_from_seed(
            config.dimension, config.num_simhash_projections, current_seed
        )
        all_sketches = all_points @ simhash_matrix

        if use_identity_proj:
            projected_points = all_points
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(
                config.dimension, projection_dim, current_seed
            )
            projected_points = all_points @ ams_matrix
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        partition_indices = _compute_partition_indices_numba(all_sketches, config.num_simhash_projections)

        rep_fde_out, partition_counts = _aggregate_avg_batch_numba_parallel(
            projected_points, doc_boundaries, partition_indices,
            num_docs, num_partitions, projection_dim
        )

        if config.fill_empty_partitions:
            _fill_empty_partitions_numba(
                rep_fde_out, partition_counts, doc_boundaries, all_sketches,
                projected_points, num_docs, num_partitions, projection_dim, config.num_simhash_projections
            )

        rep_output_start = rep_num * num_partitions * projection_dim
        out_fdes[:, rep_output_start : rep_output_start + num_partitions * projection_dim] = rep_fde_out.reshape(num_docs, -1)

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        indices, signs = _generate_count_sketch_params_numba(
            config.seed, final_fde_dim, config.final_projection_dimension
        )
        out_fdes = _apply_count_sketch_batch_numba(out_fdes, indices, signs, config.final_projection_dimension)

    return out_fdes


def generate_query_fde_batch(
    query_embeddings_list: list[np.ndarray], config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    num_queries = len(query_embeddings_list)
    if num_queries == 0:
        return np.array([])

    if config.fill_empty_partitions:
        raise ValueError(
            "Query FDE batch does not support 'fill_empty_partitions'."
        )

    valid_queries = []
    for i, q in enumerate(query_embeddings_list):
        if q.ndim != 2:
            continue
        if q.shape[1] != config.dimension:
            raise ValueError(
                f"Query {i} has incorrect dimension: expected {config.dimension}, got {q.shape[1]}"
            )
        if q.shape[0] == 0:
            continue
        valid_queries.append(q)

    if not valid_queries:
        return np.array([])

    num_queries = len(valid_queries)
    query_embeddings_list = valid_queries

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    if use_identity_proj:
        projection_dim = config.dimension
    else:
        if not config.projection_dimension or config.projection_dimension <= 0:
            raise ValueError(
                "A positive projection_dimension is required for non-identity projections."
            )
        projection_dim = config.projection_dimension

    num_partitions = 2 ** config.num_simhash_projections

    query_lengths = np.array([len(q) for q in query_embeddings_list], dtype=np.int32)
    query_boundaries = np.insert(np.cumsum(query_lengths), 0, 0)
    all_points = np.ascontiguousarray(np.vstack(query_embeddings_list).astype(np.float32))

    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fdes = np.zeros((num_queries, final_fde_dim), dtype=np.float32)

    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num

        simhash_matrix = _simhash_matrix_from_seed(
            config.dimension, config.num_simhash_projections, current_seed
        )
        all_sketches = all_points @ simhash_matrix
        
        if use_identity_proj:
            projected_points = all_points
        else:
            ams_matrix = _ams_projection_matrix_from_seed(
                config.dimension, projection_dim, current_seed
            )
            projected_points = all_points @ ams_matrix

        partition_indices = _compute_partition_indices_numba(all_sketches, config.num_simhash_projections)
        
        rep_fde_out = _aggregate_sum_batch_numba_parallel(
            projected_points, query_boundaries, partition_indices, 
            num_queries, num_partitions, projection_dim
        )

        rep_output_start = rep_num * num_partitions * projection_dim
        out_fdes[:, rep_output_start : rep_output_start + num_partitions * projection_dim] = rep_fde_out

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        indices, signs = _generate_count_sketch_params_numba(
            config.seed, final_fde_dim, config.final_projection_dimension
        )
        out_fdes = _apply_count_sketch_batch_numba(out_fdes, indices, signs, config.final_projection_dimension)

    return out_fdes


def _generate_fde_internal(
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

    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return generate_query_fde_batch([point_cloud], config)[0]
    else:
        return generate_document_fde_batch([point_cloud], config)[0]


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
    elif config.encoding_type == EncodingType.AVERAGE:
        return generate_document_fde(point_cloud, config)
    else:
        raise ValueError(f"Unsupported encoding type in config: {config.encoding_type}")


if __name__ == "__main__":
    print(f"\n{'=' * 20} SCENARIO 1: Basic FDE Generation (NUMBA) {'=' * 20}")
    base_config = FixedDimensionalEncodingConfig(
        dimension=128, num_repetitions=2, num_simhash_projections=4, seed=42
    )
    query_data = np.random.randn(32, base_config.dimension).astype(np.float32)
    doc_data = np.random.randn(80, base_config.dimension).astype(np.float32)

    # Note: first call triggers Numba compilation, so compilation time will be included
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
    query_data_list = [np.random.randn(np.random.randint(10, 50), base_config.dimension).astype(np.float32) for _ in range(5)]
    query_fde_batch = generate_query_fde_batch(query_data_list, base_config)
    
    print(f"Batch Output shape: {query_fde_batch.shape}")
    assert query_fde_batch.shape == (5, expected_dim)

    print("\nAll test scenarios completed successfully.")
