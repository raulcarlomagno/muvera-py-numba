from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

import fde_generator as original
import fde_generator_numba as numba_impl

BENCHMARK_CONFIG_SEED = 42


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
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols)).astype(np.float32)


def _docs(seed: int, count: int, length: int, dimension: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(length, dimension)).astype(np.float32) for _ in range(count)]


BENCHMARK_CASES: dict[str, dict[str, Any]] = {
    "query_single_small": {
        "kind": "query_single",
        "size_label": "24 x 384",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 1_001,
        "length": 24,
    },
    "query_single_medium": {
        "kind": "query_single",
        "size_label": "96 x 512",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 1_002,
        "length": 96,
    },
    "query_single_large": {
        "kind": "query_single",
        "size_label": "384 x 768",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 1_003,
        "length": 384,
    },
    "document_single_small": {
        "kind": "document_single",
        "size_label": "32 x 384",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 2_001,
        "length": 32,
    },
    "document_single_medium": {
        "kind": "document_single",
        "size_label": "128 x 512",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 2_002,
        "length": 128,
    },
    "document_single_large": {
        "kind": "document_single",
        "size_label": "512 x 768",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 2_003,
        "length": 512,
    },
    "document_batch_small": {
        "kind": "document_batch",
        "size_label": "100 docs x 128 x 384",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 3_001,
        "doc_count": 100,
        "doc_length": 128,
    },
    "document_batch_medium": {
        "kind": "document_batch",
        "size_label": "2500 docs x 512 x 512",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 3_002,
        "doc_count": 2500,
        "doc_length": 512,
    },
    "document_batch_large": {
        "kind": "document_batch",
        "size_label": "10000 docs x 1024 x 768",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 3_003,
        "doc_count": 10000,
        "doc_length": 1024,
    },
    "query_batch_small": {
        "kind": "query_batch",
        "size_label": "100 queries x 128 x 384",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 4_001,
        "query_count": 100,
        "query_length": 128,
    },
    "query_batch_medium": {
        "kind": "query_batch",
        "size_label": "2500 queries x 512 x 512",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 4_002,
        "query_count": 2500,
        "query_length": 512,
    },
    "query_batch_large": {
        "kind": "query_batch",
        "size_label": "10000 queries x 1024 x 768",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=3,
            num_simhash_projections=5,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 4_003,
        "query_count": 10000,
        "query_length": 1024,
    },
}


def _build_case_inputs(case_name: str) -> tuple[str, str, Any, Any, numba_impl.FixedDimensionalEncodingConfig]:
    case = BENCHMARK_CASES[case_name]
    config = case["config"]
    original_config = _to_original_config(config)
    kind = case["kind"]

    if kind == "query_single":
        point_cloud = _matrix(case["data_seed"], case["length"], config.dimension)
        return kind, case["size_label"], point_cloud, point_cloud, config

    if kind == "document_single":
        point_cloud = _matrix(case["data_seed"], case["length"], config.dimension)
        return kind, case["size_label"], point_cloud, point_cloud, config

    if kind == "document_batch":
        docs = _docs(
            case["data_seed"], case["doc_count"], case["doc_length"], config.dimension
        )
        return kind, case["size_label"], docs, docs, config

    if kind == "query_batch":
        queries = _docs(
            case["data_seed"],
            case["query_count"],
            case["query_length"],
            config.dimension,
        )
        return kind, case["size_label"], queries, queries, config

    raise ValueError(f"Unsupported benchmark case kind: {kind}")


def _get_runners(
    kind: str,
    original_inputs: Any,
    numba_inputs: Any,
    config: numba_impl.FixedDimensionalEncodingConfig,
) -> tuple[Callable[[], np.ndarray], Callable[[], np.ndarray]]:
    original_config = _to_original_config(config)

    if kind == "query_single":
        return (
            lambda: original.generate_query_fde(original_inputs, original_config),
            lambda: numba_impl.generate_query_fde(numba_inputs, config),
        )

    if kind == "document_single":
        return (
            lambda: original.generate_document_fde(original_inputs, original_config),
            lambda: numba_impl.generate_document_fde(numba_inputs, config),
        )

    if kind == "document_batch":
        return (
            lambda: original.generate_document_fde_batch(original_inputs, original_config),
            lambda: numba_impl.generate_document_fde_batch(numba_inputs, config),
        )

    if kind == "query_batch":
        return (
            lambda: np.vstack(
                [
                    original.generate_query_fde(query, original_config)
                    for query in original_inputs
                ]
            ),
            lambda: numba_impl.generate_query_fde_batch(numba_inputs, config),
        )

    raise ValueError(f"Unsupported runner kind: {kind}")


def _measure_once(fn: Callable[[], np.ndarray]) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, result


def _measure_median(fn: Callable[[], np.ndarray], repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        elapsed, _ = _measure_once(fn)
        timings.append(elapsed)
    return statistics.median(timings)


def _validate_outputs(case_name: str) -> None:
    kind, _, original_inputs, numba_inputs, config = _build_case_inputs(case_name)
    original_runner, numba_runner = _get_runners(kind, original_inputs, numba_inputs, config)
    expected = original_runner()
    actual = numba_runner()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _run_case(case_name: str, repeats: int, validate: bool) -> dict[str, Any]:
    kind, size_label, original_inputs, numba_inputs, config = _build_case_inputs(case_name)
    original_runner, numba_runner = _get_runners(kind, original_inputs, numba_inputs, config)

    if validate:
        expected = original_runner()
        actual = numba_runner()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    original_time = _measure_median(original_runner, repeats)
    numba_cold_time, _ = _measure_once(numba_runner)
    _ = numba_runner()
    numba_warm_time = _measure_median(numba_runner, repeats)

    return {
        "case": case_name,
        "kind": kind,
        "size": size_label,
        "repeats": repeats,
        "original_s": original_time,
        "numba_cold_s": numba_cold_time,
        "numba_warm_s": numba_warm_time,
        "speedup": original_time / numba_warm_time if numba_warm_time else float("inf"),
    }


def _run_case_in_subprocess(case_name: str, repeats: int, validate: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        __file__,
        "--worker",
        case_name,
        "--repeats",
        str(repeats),
    ]
    if validate:
        command.append("--validate")

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _print_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "case",
        "size",
        "repeats",
        "original_ms",
        "numba_cold_ms",
        "numba_warm_ms",
        "speedup",
    ]
    rows = [
        [
            result["case"],
            result["size"],
            str(result["repeats"]),
            _format_ms(result["original_s"]),
            _format_ms(result["numba_cold_s"]),
            _format_ms(result["numba_warm_s"]),
            f"{result['speedup']:.2f}x",
        ]
        for result in results
    ]

    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the original and Numba FDE implementations."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of timing repetitions per case for original and warm Numba runs.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=sorted(BENCHMARK_CASES),
        help="Optional subset of benchmark cases to execute.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the numerical equivalence check before timing.",
    )
    parser.add_argument(
        "--worker",
        choices=sorted(BENCHMARK_CASES),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    if args.worker:
        result = _run_case(args.worker, args.repeats, args.validate)
        print(json.dumps(result))
        return

    selected_cases = args.cases or list(BENCHMARK_CASES.keys())
    validate = not args.skip_validate

    results = [
        _run_case_in_subprocess(case_name, args.repeats, validate)
        for case_name in selected_cases
    ]

    print("FDE speed benchmark")
    print("cold = first Numba call in a fresh worker process")
    print("warm = median after explicit warmup in the same worker process")
    if validate:
        print("validation = enabled")
    else:
        print("validation = skipped")
    print()
    _print_table(results)


if __name__ == "__main__":
    main()
