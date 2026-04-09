from dataclasses import replace

import numpy as np
import pytest

import fde_generator as original
import fde_generator_numba as numba_impl


def _to_original_config(
    config: numba_impl.FixedDimensionalEncodingConfig,
) -> original.FixedDimensionalEncodingConfig:
    return original.FixedDimensionalEncodingConfig(
        dimension=config.dimension,
        num_repetitions=config.num_repetitions,
        num_simhash_projections=config.num_simhash_projections,
        seed=config.seed,
        encoding_type=original.EncodingType[config.encoding_type.name],
        projection_type=original.ProjectionType[config.projection_type.name],
        projection_dimension=config.projection_dimension,
        fill_empty_partitions=config.fill_empty_partitions,
        final_projection_dimension=config.final_projection_dimension,
    )


def _matrix(seed: int, rows: int, cols: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)


def _docs(seed: int, lengths: list[int], dimension: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(length, dimension)).astype(np.float32) for length in lengths]


def _assert_close(left: np.ndarray, right: np.ndarray) -> None:
    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


QUERY_CASES = [
    (
        "identity",
        replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=16,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=11,
        ),
        24,
        1101,
    ),
    (
        "ams",
        replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=16,
            num_repetitions=3,
            num_simhash_projections=4,
            seed=21,
            projection_type=numba_impl.ProjectionType.AMS_SKETCH,
            projection_dimension=8,
        ),
        18,
        1201,
    ),
    (
        "final_projection",
        replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=16,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=31,
            final_projection_dimension=32,
        ),
        20,
        1301,
    ),
]


DOCUMENT_CASES = [
    (
        "identity",
        replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=16,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=41,
            fill_empty_partitions=True,
        ),
        [5, 8, 3],
        2101,
    ),
    (
        "ams",
        replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=16,
            num_repetitions=2,
            num_simhash_projections=4,
            seed=51,
            projection_type=numba_impl.ProjectionType.AMS_SKETCH,
            projection_dimension=8,
            fill_empty_partitions=True,
        ),
        [3, 4, 2],
        2201,
    ),
    (
        "final_projection",
        replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=16,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=61,
            final_projection_dimension=24,
        ),
        [6, 5],
        2301,
    ),
]


@pytest.mark.parametrize("case_name, config, length, data_seed", QUERY_CASES)
def test_query_fde_matches_original(
    case_name: str, config, length: int, data_seed: int
) -> None:
    query = _matrix(data_seed, length, config.dimension)
    expected = original.generate_query_fde(query, _to_original_config(config))
    actual = numba_impl.generate_query_fde(query, config)
    _assert_close(actual, expected)


@pytest.mark.parametrize("case_name, config, length, data_seed", QUERY_CASES)
def test_generate_fde_dispatch_matches_query_path(
    case_name: str, config, length: int, data_seed: int
) -> None:
    query = _matrix(data_seed, length, config.dimension)
    expected = original.generate_query_fde(query, _to_original_config(config))
    actual = numba_impl.generate_fde(query, config)
    _assert_close(actual, expected)


@pytest.mark.parametrize("case_name, config, length, data_seed", QUERY_CASES)
def test_query_single_matches_batch_singleton(
    case_name: str, config, length: int, data_seed: int
) -> None:
    query = _matrix(data_seed, length, config.dimension)
    expected = numba_impl.generate_query_fde_batch([query], config)[0]
    actual = numba_impl.generate_query_fde(query, config)
    _assert_close(actual, expected)


@pytest.mark.parametrize("case_name, config, lengths, data_seed", DOCUMENT_CASES)
def test_document_fde_matches_original(
    case_name: str, config, lengths: list[int], data_seed: int
) -> None:
    docs = _docs(data_seed, lengths, config.dimension)
    original_config = _to_original_config(config)

    for doc in docs:
        expected = original.generate_document_fde(doc, original_config)
        actual = numba_impl.generate_document_fde(doc, config)
        _assert_close(actual, expected)


@pytest.mark.parametrize("case_name, config, lengths, data_seed", DOCUMENT_CASES)
def test_document_single_matches_batch_singleton(
    case_name: str, config, lengths: list[int], data_seed: int
) -> None:
    docs = _docs(data_seed, lengths, config.dimension)
    for doc in docs:
        expected = numba_impl.generate_document_fde_batch([doc], config)[0]
        actual = numba_impl.generate_document_fde(doc, config)
        _assert_close(actual, expected)


@pytest.mark.parametrize("case_name, config, lengths, data_seed", DOCUMENT_CASES)
def test_document_batch_matches_original(
    case_name: str, config, lengths: list[int], data_seed: int
) -> None:
    docs = _docs(data_seed, lengths, config.dimension)
    expected = original.generate_document_fde_batch(docs, _to_original_config(config))
    actual = numba_impl.generate_document_fde_batch(docs, config)
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    "case_name, config, lengths, data_seed",
    [
        (
            "query_batch_identity",
            replace(
                numba_impl.FixedDimensionalEncodingConfig(),
                dimension=16,
                num_repetitions=2,
                num_simhash_projections=3,
                seed=71,
            ),
            [4, 7, 5],
            3101,
        ),
        (
            "query_batch_ams",
            replace(
                numba_impl.FixedDimensionalEncodingConfig(),
                dimension=16,
                num_repetitions=3,
                num_simhash_projections=4,
                seed=81,
                projection_type=numba_impl.ProjectionType.AMS_SKETCH,
                projection_dimension=8,
            ),
            [3, 6, 2, 5],
            3201,
        ),
        (
            "query_batch_final_projection",
            replace(
                numba_impl.FixedDimensionalEncodingConfig(),
                dimension=16,
                num_repetitions=2,
                num_simhash_projections=3,
                seed=91,
                final_projection_dimension=32,
            ),
            [5, 4, 6],
            3301,
        ),
    ],
)
def test_query_batch_matches_stacked_single_outputs(
    case_name: str, config, lengths: list[int], data_seed: int
) -> None:
    queries = _docs(data_seed, lengths, config.dimension)
    original_config = _to_original_config(config)
    expected = np.vstack(
        [original.generate_query_fde(query, original_config) for query in queries]
    )
    actual = numba_impl.generate_query_fde_batch(queries, config)
    _assert_close(actual, expected)
