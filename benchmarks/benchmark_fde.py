from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import re
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
BENCHMARK_NUM_REPETITIONS = 2
BENCHMARK_NUM_SIMHASH_PROJECTIONS = 4
BATCH_COUNT_SMALL = 1000
BATCH_COUNT_MEDIUM = 5000
BATCH_COUNT_LARGE = 10000
BATCH_LENGTH_MEAN = 50.0
BATCH_LENGTH_STD = 15.0
BATCH_LENGTH_MIN = 1
BATCH_LENGTH_MAX = 100


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


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


def _sample_batch_lengths(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lengths = np.rint(rng.normal(BATCH_LENGTH_MEAN, BATCH_LENGTH_STD, size=count))
    lengths = np.clip(lengths, BATCH_LENGTH_MIN, BATCH_LENGTH_MAX)
    return lengths.astype(np.int32)


def _docs(seed: int, lengths: np.ndarray, dimension: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.normal(size=(int(length), dimension)).astype(np.float32)
        for length in lengths
    ]


def _available_physical_memory_bytes() -> int | None:
    system = platform.system()

    if system == "Windows":
        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            if page_size > 0 and available_pages > 0:
                return page_size * available_pages
        except (AttributeError, OSError, ValueError):
            pass

    if system == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        except (FileNotFoundError, OSError, ValueError, IndexError):
            pass

    if system == "Darwin":
        try:
            completed = subprocess.run(
                ["vm_stat"],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = completed.stdout.splitlines()
            page_size_match = re.search(r"page size of (\d+) bytes", lines[0])
            if not page_size_match:
                return None

            page_size = int(page_size_match.group(1))
            page_counts: dict[str, int] = {}
            for line in lines[1:]:
                match = re.match(r"^(.*?):\s+(\d+)\.$", line.strip())
                if match:
                    page_counts[match.group(1)] = int(match.group(2))

            available_pages = (
                page_counts.get("Pages free", 0)
                + page_counts.get("Pages inactive", 0)
                + page_counts.get("Pages speculative", 0)
            )
            if available_pages > 0:
                return available_pages * page_size
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            pass

    return None


def _estimate_case_input_bytes(case: dict[str, Any]) -> int:
    float32_size = np.dtype(np.float32).itemsize
    kind = case["kind"]

    if kind in {"query_single", "document_single"}:
        return case["length"] * case["config"].dimension * float32_size

    if kind == "document_batch":
        lengths = _batch_lengths_for_case(case)
        return int(lengths.sum()) * case["config"].dimension * float32_size

    if kind == "query_batch":
        lengths = _batch_lengths_for_case(case)
        return int(lengths.sum()) * case["config"].dimension * float32_size

    raise ValueError(f"Unsupported benchmark case kind: {kind}")


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def _preflight_case_memory(case_name: str) -> str | None:
    case = BENCHMARK_CASES[case_name]
    required_bytes = _estimate_case_input_bytes(case)
    available_bytes = _available_physical_memory_bytes()

    if available_bytes is None:
        return None

    # Reserve generous headroom because both implementations allocate extra
    # arrays for stacking, projections, aggregation, and outputs.
    safe_budget = int(available_bytes * 0.5)
    if required_bytes > safe_budget:
        return (
            f"estimated raw input {_format_gib(required_bytes)} exceeds safe memory "
            f"budget {_format_gib(safe_budget)}"
        )
    return None


BENCHMARK_CASES: dict[str, dict[str, Any]] = {
    "query_single_small": {
        "kind": "query_single",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 1_001,
        "length": 24,
    },
    "query_single_medium": {
        "kind": "query_single",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 1_002,
        "length": 96,
    },
    "query_single_large": {
        "kind": "query_single",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 1_003,
        "length": 384,
    },
    "document_single_small": {
        "kind": "document_single",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 2_001,
        "length": 32,
    },
    "document_single_medium": {
        "kind": "document_single",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 2_002,
        "length": 128,
    },
    "document_single_large": {
        "kind": "document_single",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 2_003,
        "length": 512,
    },
    "document_batch_small": {
        "kind": "document_batch",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 3_001,
        "doc_count": BATCH_COUNT_SMALL,
        "length_mean": BATCH_LENGTH_MEAN,
        "length_std": BATCH_LENGTH_STD,
        "min_length": BATCH_LENGTH_MIN,
        "max_length": BATCH_LENGTH_MAX,
    },
    "document_batch_medium": {
        "kind": "document_batch",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 3_002,
        "doc_count": BATCH_COUNT_MEDIUM,
        "length_mean": BATCH_LENGTH_MEAN,
        "length_std": BATCH_LENGTH_STD,
        "min_length": BATCH_LENGTH_MIN,
        "max_length": BATCH_LENGTH_MAX,
    },
    "document_batch_large": {
        "kind": "document_batch",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
            fill_empty_partitions=True,
        ),
        "data_seed": 3_003,
        "doc_count": BATCH_COUNT_LARGE,
        "length_mean": BATCH_LENGTH_MEAN,
        "length_std": BATCH_LENGTH_STD,
        "min_length": BATCH_LENGTH_MIN,
        "max_length": BATCH_LENGTH_MAX,
    },
    "query_batch_small": {
        "kind": "query_batch",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=384,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 4_001,
        "query_count": BATCH_COUNT_SMALL,
        "length_mean": BATCH_LENGTH_MEAN,
        "length_std": BATCH_LENGTH_STD,
        "min_length": BATCH_LENGTH_MIN,
        "max_length": BATCH_LENGTH_MAX,
    },
    "query_batch_medium": {
        "kind": "query_batch",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=512,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 4_002,
        "query_count": BATCH_COUNT_MEDIUM,
        "length_mean": BATCH_LENGTH_MEAN,
        "length_std": BATCH_LENGTH_STD,
        "min_length": BATCH_LENGTH_MIN,
        "max_length": BATCH_LENGTH_MAX,
    },
    "query_batch_large": {
        "kind": "query_batch",
        "config": replace(
            numba_impl.FixedDimensionalEncodingConfig(),
            dimension=768,
            num_repetitions=BENCHMARK_NUM_REPETITIONS,
            num_simhash_projections=BENCHMARK_NUM_SIMHASH_PROJECTIONS,
            seed=BENCHMARK_CONFIG_SEED,
        ),
        "data_seed": 4_003,
        "query_count": BATCH_COUNT_LARGE,
        "length_mean": BATCH_LENGTH_MEAN,
        "length_std": BATCH_LENGTH_STD,
        "min_length": BATCH_LENGTH_MIN,
        "max_length": BATCH_LENGTH_MAX,
    },
}


def _batch_lengths_for_case(case: dict[str, Any]) -> np.ndarray:
    if case["kind"] == "document_batch":
        return _sample_batch_lengths(case["data_seed"], case["doc_count"])

    if case["kind"] == "query_batch":
        return _sample_batch_lengths(case["data_seed"], case["query_count"])

    raise ValueError(f"Unsupported batch case kind: {case['kind']}")


def _case_size_label(case: dict[str, Any]) -> str:
    dimension = case["config"].dimension
    kind = case["kind"]

    if kind in {"query_single", "document_single"}:
        return f"{case['length']} x {dimension}"

    if kind == "document_batch":
        lengths = _batch_lengths_for_case(case)
        return (
            f"{case['doc_count']} docs x len[{int(lengths.min())}-{int(lengths.max())}] "
            f"avg={lengths.mean():.1f} x {dimension}"
        )

    if kind == "query_batch":
        lengths = _batch_lengths_for_case(case)
        return (
            f"{case['query_count']} queries x len[{int(lengths.min())}-{int(lengths.max())}] "
            f"avg={lengths.mean():.1f} x {dimension}"
        )

    raise ValueError(f"Unsupported benchmark case kind: {kind}")


def _build_case_inputs(
    case_name: str,
) -> tuple[str, str, Any, Any, numba_impl.FixedDimensionalEncodingConfig]:
    case = BENCHMARK_CASES[case_name]
    config = case["config"]
    kind = case["kind"]
    size_label = _case_size_label(case)

    if kind == "query_single":
        point_cloud = _matrix(case["data_seed"], case["length"], config.dimension)
        return kind, size_label, point_cloud, point_cloud, config

    if kind == "document_single":
        point_cloud = _matrix(case["data_seed"], case["length"], config.dimension)
        return kind, size_label, point_cloud, point_cloud, config

    if kind == "document_batch":
        lengths = _batch_lengths_for_case(case)
        docs = _docs(case["data_seed"] + 1, lengths, config.dimension)
        return kind, size_label, docs, docs, config

    if kind == "query_batch":
        lengths = _batch_lengths_for_case(case)
        queries = _docs(case["data_seed"] + 1, lengths, config.dimension)
        return kind, size_label, queries, queries, config

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
            lambda: original.generate_document_fde_batch(
                original_inputs, original_config
            ),
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


def _cleanup_memory(*objects: Any) -> None:
    # Drop references to large arrays/lists eagerly. NumPy memory is released
    # when references disappear; the subprocess boundary guarantees the OS
    # reclaims everything when each worker exits.
    gc.collect()


def _run_case(case_name: str, repeats: int, validate: bool) -> dict[str, Any]:
    skip_reason = _preflight_case_memory(case_name)
    if skip_reason:
        case = BENCHMARK_CASES[case_name]
        return {
            "case": case_name,
            "kind": case["kind"],
            "size": _case_size_label(case),
            "repeats": repeats,
            "status": "skipped",
            "reason": skip_reason,
        }

    kind, size_label, original_inputs, numba_inputs, config = _build_case_inputs(
        case_name
    )
    original_runner, numba_runner = _get_runners(
        kind, original_inputs, numba_inputs, config
    )

    try:
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
            "status": "ok",
            "original_s": original_time,
            "numba_cold_s": numba_cold_time,
            "numba_warm_s": numba_warm_time,
            "speedup": original_time / numba_warm_time
            if numba_warm_time
            else float("inf"),
        }
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as exc:
        return {
            "case": case_name,
            "kind": kind,
            "size": size_label,
            "repeats": repeats,
            "status": "skipped",
            "reason": f"memory allocation failed: {exc}",
        }
    finally:
        del original_inputs
        del numba_inputs
        del original_runner
        del numba_runner
        del config
        _cleanup_memory()


def _run_case_in_subprocess(
    case_name: str, repeats: int, validate: bool
) -> dict[str, Any]:
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

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip() or "<empty>"
        stderr = completed.stderr.strip() or "<empty>"
        raise RuntimeError(
            f"Benchmark worker failed for '{case_name}'.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    return json.loads(completed.stdout)


def _print_progress_start(
    index: int, total: int, case_name: str, size_label: str
) -> None:
    print(f"[{index}/{total}] Running {case_name} ({size_label})...", flush=True)


def _print_progress_end(result: dict[str, Any]) -> None:
    if result["status"] == "ok":
        print(
            f"    completed: original={_format_ms(result['original_s'])} ms, "
            f"numba_warm={_format_ms(result['numba_warm_s'])} ms, "
            f"speedup={result['speedup']:.2f}x",
            flush=True,
        )
    else:
        print(f"    skipped: {result['reason']}", flush=True)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _print_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "case",
        "size",
        "repeats",
        "status",
        "original_ms",
        "numba_cold_ms",
        "numba_warm_ms",
        "speedup",
        "notes",
    ]
    rows = []
    for result in results:
        if result["status"] == "ok":
            rows.append(
                [
                    result["case"],
                    result["size"],
                    str(result["repeats"]),
                    result["status"],
                    _format_ms(result["original_s"]),
                    _format_ms(result["numba_cold_s"]),
                    _format_ms(result["numba_warm_s"]),
                    f"{result['speedup']:.2f}x",
                    "",
                ]
            )
        else:
            rows.append(
                [
                    result["case"],
                    result["size"],
                    str(result["repeats"]),
                    result["status"],
                    "-",
                    "-",
                    "-",
                    "-",
                    result["reason"],
                ]
            )

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
        "--worker",
        choices=sorted(BENCHMARK_CASES),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the numerical equivalence check before timing each case.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    if args.worker:
        try:
            result = _run_case(args.worker, args.repeats, args.validate)
            print(json.dumps(result))
            return
        finally:
            _cleanup_memory()

    selected_cases = args.cases or list(BENCHMARK_CASES.keys())
    validate = args.validate

    results = []
    total_cases = len(selected_cases)
    for index, case_name in enumerate(selected_cases, start=1):
        size_label = _case_size_label(BENCHMARK_CASES[case_name])
        _print_progress_start(index, total_cases, case_name, size_label)
        result = _run_case_in_subprocess(case_name, args.repeats, validate)
        _print_progress_end(result)
        results.append(result)
        print(flush=True)

    print("FDE speed benchmark")
    print("cold = first Numba call in a fresh worker process")
    print("warm = median after explicit warmup in the same worker process")
    if validate:
        print("validation = enabled")
    else:
        print("validation = skipped")
    print()
    _print_table(results)
    _cleanup_memory(results)


if __name__ == "__main__":
    main()
