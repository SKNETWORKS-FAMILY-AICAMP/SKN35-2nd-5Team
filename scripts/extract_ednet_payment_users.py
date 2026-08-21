#!/usr/bin/env python3
"""Extract complete logs for every EdNet KT4 user with pay/refund activity.

The job is intentionally multi-pass and memory bounded:

1. Scan each source CSV using only its ``action_type`` and ``timestamp`` fields.
2. Print exact selected-user/row counts and a conservative output-size estimate.
3. Re-read only selected users, sort one user's rows at a time, and stream them
   into a single CSV while adding ``user_id`` and UTC ``datetime``.
4. Stream the completed output once more to verify user/row preservation.

No source KT4 file is changed, and no churn label or model feature is created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CANDIDATES = (PROJECT_ROOT / "KT4", PROJECT_ROOT / "data" / "KT4")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ednet_payment_users.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "ednet_payment_user_summary.csv"
SUMMARY_FIELDS = [
    "user_id",
    "has_pay",
    "has_refund",
    "pay_count",
    "refund_count",
    "has_pay_then_refund",
    "refund_before_first_pay",
    "first_timestamp",
    "last_timestamp",
    "event_count",
    "first_pay_timestamp",
    "first_refund_timestamp",
    "last_pay_timestamp",
    "last_refund_timestamp",
]
MANIFEST_FIELDS = ["relative_path", *SUMMARY_FIELDS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all logs for KT4 users containing pay or refund events."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="KT4 folder (default: auto-detect project/KT4 or project/data/KT4)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Report file progress every N files (default: 1000)",
    )
    parser.add_argument(
        "--validation-progress-every",
        type=int,
        default=5_000_000,
        help="Report validation progress every N output rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development-only input file limit; omit for a full extraction",
    )
    return parser.parse_args()


def resolve_input(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"KT4 input directory not found: {path}")
        return path
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    attempted = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"KT4 directory was not found. Tried: {attempted}")


def timestamp_seconds_and_unit(raw: str) -> tuple[float | None, str]:
    text = raw.strip()
    if not text:
        return None, "invalid"
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"
    if not math.isfinite(value):
        return None, "invalid"
    magnitude = abs(value)
    if magnitude >= 1e17:
        seconds, unit = value / 1e9, "ns"
    elif magnitude >= 1e14:
        seconds, unit = value / 1e6, "us"
    elif magnitude >= 1e11:
        seconds, unit = value / 1e3, "ms"
    else:
        seconds, unit = value, "s"
    try:
        datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None, "invalid"
    return seconds, unit


def utc_datetime(raw_timestamp: str) -> str:
    seconds, _unit = timestamp_seconds_and_unit(raw_timestamp)
    if seconds is None:
        return ""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


def normalized_header(header: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, name in enumerate(header):
        normalized = name.strip().casefold()
        if normalized in result:
            raise ValueError(f"duplicate column after case normalization: {name!r}")
        result[normalized] = index
    return result


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_bytes(size: float) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    return f"{size / 1024**2:.2f} MiB"


def progress_line(
    phase: str,
    current: int,
    total: int,
    started: float,
    selected: int | None = None,
) -> str:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed else 0.0
    remaining = (total - current) / rate if rate else float("inf")
    selected_text = f" | selected={selected:,}" if selected is not None else ""
    return (
        f"{phase}: {current:,} / {total:,}{selected_text} | "
        f"{rate:.1f} files/s | ETA {format_duration(remaining)}"
    )


def scan_file(path: Path, user_id: str) -> tuple[dict[str, Any], list[str], Counter[str]]:
    """First-pass scan: inspect timestamp/action only and return a compact summary."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("empty CSV (missing header)") from exc
        lookup = normalized_header(header)
        missing = {"timestamp", "action_type"} - set(lookup)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        timestamp_index = lookup["timestamp"]
        action_index = lookup["action_type"]
        required_index = max(timestamp_index, action_index)

        event_count = 0
        pay_count = 0
        refund_count = 0
        all_times: list[tuple[float, str]] = []
        pay_times: list[tuple[float, str]] = []
        refund_times: list[tuple[float, str]] = []
        unit_counts: Counter[str] = Counter()

        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) <= required_index:
                raise ValueError(
                    f"row {row_number} has {len(row)} columns; expected at least "
                    f"{required_index + 1}"
                )
            event_count += 1
            action = row[action_index].strip().casefold()
            raw_timestamp = row[timestamp_index].strip()
            seconds, unit = timestamp_seconds_and_unit(raw_timestamp)
            unit_counts[unit] += 1
            if seconds is not None:
                timed = (seconds, raw_timestamp)
                all_times.append(timed)
                if action == "pay":
                    pay_times.append(timed)
                elif action == "refund":
                    refund_times.append(timed)
            if action == "pay":
                pay_count += 1
            elif action == "refund":
                refund_count += 1

    all_times.sort(key=lambda value: value[0])
    pay_times.sort(key=lambda value: value[0])
    refund_times.sort(key=lambda value: value[0])
    has_pay_then_refund = int(
        bool(pay_times)
        and bool(refund_times)
        and any(refund[0] >= pay_times[0][0] for refund in refund_times)
    )
    refund_before_first_pay = int(
        bool(pay_times) and bool(refund_times) and refund_times[0][0] < pay_times[0][0]
    )

    def raw_at(values: list[tuple[float, str]], index: int) -> str:
        return values[index][1] if values else ""

    summary = {
        "user_id": user_id,
        "has_pay": int(pay_count > 0),
        "has_refund": int(refund_count > 0),
        "pay_count": pay_count,
        "refund_count": refund_count,
        "has_pay_then_refund": has_pay_then_refund,
        "refund_before_first_pay": refund_before_first_pay,
        "first_timestamp": raw_at(all_times, 0),
        "last_timestamp": raw_at(all_times, -1),
        "event_count": event_count,
        "first_pay_timestamp": raw_at(pay_times, 0),
        "first_refund_timestamp": raw_at(refund_times, 0),
        "last_pay_timestamp": raw_at(pay_times, -1),
        "last_refund_timestamp": raw_at(refund_times, -1),
    }
    return summary, header, unit_counts


def read_full_user_rows(
    path: Path,
) -> tuple[list[str], list[tuple[tuple[int, float, int], list[str]]]]:
    """Read and timestamp-sort one selected user's complete rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("empty CSV (missing header)") from exc
        lookup = normalized_header(header)
        if "timestamp" not in lookup:
            raise ValueError("missing required column: timestamp")
        timestamp_index = lookup["timestamp"]
        rows: list[tuple[tuple[int, float, int], list[str]]] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) <= timestamp_index:
                raise ValueError(f"row {row_number} is missing timestamp")
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            elif len(row) > len(header):
                raise ValueError(
                    f"row {row_number} has {len(row)} values but header has {len(header)}"
                )
            seconds, _unit = timestamp_seconds_and_unit(row[timestamp_index])
            sort_key = (
                0 if seconds is not None else 1,
                seconds if seconds is not None else 0.0,
                row_number,
            )
            rows.append((sort_key, row))
    rows.sort(key=lambda item: item[0])
    return header, rows


def write_error_csv(path: Path, failures: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "source_file", "error_type", "error_message"]
        )
        writer.writeheader()
        writer.writerows(failures)


def validate_outputs(
    output_path: Path,
    summary_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    """Re-read final artifacts and verify all-user row and payment preservation."""
    expected_rows: dict[str, int] = {}
    expected_payment: dict[str, tuple[bool, bool]] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            expected_rows[row["user_id"]] = int(row["event_count"])
            expected_payment[row["user_id"]] = (
                row["has_pay"] == "1",
                row["has_refund"] == "1",
            )

    actual_rows: Counter[str] = Counter()
    actual_payment: dict[str, list[bool]] = {}
    total_rows = 0
    started = time.monotonic()
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"user_id", "action_type"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("final output is missing user_id or action_type")
        for row in reader:
            total_rows += 1
            user_id = row["user_id"]
            actual_rows[user_id] += 1
            flags = actual_payment.setdefault(user_id, [False, False])
            action = row["action_type"].strip().casefold()
            if action == "pay":
                flags[0] = True
            elif action == "refund":
                flags[1] = True
            if progress_every > 0 and total_rows % progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"Validating output: {total_rows:,} rows | "
                    f"{total_rows / elapsed:,.0f} rows/s",
                    flush=True,
                )

    row_mismatches = {
        user_id: {"expected": expected, "actual": actual_rows.get(user_id, 0)}
        for user_id, expected in expected_rows.items()
        if actual_rows.get(user_id, 0) != expected
    }
    payment_mismatches = {
        user_id: {
            "expected": expected_payment[user_id],
            "actual": tuple(actual_payment.get(user_id, [False, False])),
        }
        for user_id in expected_payment
        if tuple(actual_payment.get(user_id, [False, False])) != expected_payment[user_id]
    }
    unexpected_users = sorted(set(actual_rows) - set(expected_rows))
    return {
        "summary_user_count": len(expected_rows),
        "integrated_unique_user_count": len(actual_rows),
        "integrated_row_count": total_rows,
        "user_counts_match": len(expected_rows) == len(actual_rows),
        "all_users_have_pay_or_refund": all(any(flags) for flags in actual_payment.values()),
        "all_user_row_counts_match": not row_mismatches and not unexpected_users,
        "all_payment_flags_match": not payment_mismatches,
        "row_mismatch_count": len(row_mismatches),
        "payment_flag_mismatch_count": len(payment_mismatches),
        "unexpected_user_count": len(unexpected_users),
        "sample_row_mismatches": dict(list(row_mismatches.items())[:20]),
        "sample_payment_mismatches": dict(list(payment_mismatches.items())[:20]),
        "sample_unexpected_users": unexpected_users[:20],
    }


def main() -> int:
    args = parse_args()
    input_dir = resolve_input(args.input)
    output_path = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    if output_path == summary_path:
        raise ValueError("--output and --summary must be different paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    output_partial = output_path.with_suffix(output_path.suffix + ".partial")
    summary_partial = summary_path.with_suffix(summary_path.suffix + ".partial")
    manifest_path = summary_path.with_suffix(summary_path.suffix + ".manifest.tmp")
    errors_path = summary_path.with_name("ednet_payment_extraction_errors.csv")
    report_path = summary_path.with_name("ednet_payment_extraction_report.json")

    print(f"Input: {input_dir}", flush=True)
    print("Discovering user CSV files...", flush=True)
    files = sorted(input_dir.rglob("*.csv"), key=lambda path: path.as_posix())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        files = files[: args.limit]
        print(f"WARNING: development limit enabled ({args.limit:,} files).", flush=True)
    total_files = len(files)
    if not total_files:
        raise FileNotFoundError(f"No CSV files found below {input_dir}")
    print(f"Found {total_files:,} user CSV files.", flush=True)

    union_columns: list[str] = []
    union_seen: set[str] = set()
    selected_users = 0
    selected_rows = 0
    selected_source_bytes = 0
    selected_user_id_characters = 0
    scan_stats: Counter[str] = Counter()
    timestamp_units: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    scan_started = time.monotonic()

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as manifest_handle:
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=MANIFEST_FIELDS)
        manifest_writer.writeheader()
        for index, path in enumerate(files, start=1):
            relative_path = path.relative_to(input_dir).as_posix()
            user_id = Path(relative_path).with_suffix("").as_posix()
            try:
                summary, header, file_units = scan_file(path, user_id)
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                failures.append(
                    {
                        "phase": "scan",
                        "source_file": relative_path,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            else:
                timestamp_units.update(file_units)
                for column in header:
                    normalized = column.strip().casefold()
                    # These two output columns are generated from filename/timestamp.
                    if normalized in {"user_id", "datetime"}:
                        continue
                    if normalized not in union_seen:
                        union_seen.add(normalized)
                        union_columns.append(column)
                has_pay = bool(summary["has_pay"])
                has_refund = bool(summary["has_refund"])
                scan_stats["pay_users"] += int(has_pay)
                scan_stats["refund_users"] += int(has_refund)
                scan_stats["pay_and_refund_users"] += int(has_pay and has_refund)
                scan_stats["pay_only_users"] += int(has_pay and not has_refund)
                scan_stats["refund_only_users"] += int(has_refund and not has_pay)
                scan_stats["pay_then_refund_users"] += int(summary["has_pay_then_refund"])
                scan_stats["refund_before_first_pay_users"] += int(
                    summary["refund_before_first_pay"]
                )
                if has_pay or has_refund:
                    selected_users += 1
                    selected_rows += int(summary["event_count"])
                    selected_source_bytes += path.stat().st_size
                    selected_user_id_characters += len(user_id.encode("utf-8"))
                    manifest_writer.writerow(
                        {"relative_path": relative_path, **summary}
                    )
            if (
                index == 1
                or index == total_files
                or (args.progress_every > 0 and index % args.progress_every == 0)
            ):
                print(
                    progress_line(
                        "Scanning KT4 users",
                        index,
                        total_files,
                        scan_started,
                        selected_users,
                    ),
                    flush=True,
                )

    average_user_id_bytes = (
        selected_user_id_characters / selected_users if selected_users else 0
    )
    # Source byte total is a strong baseline; add user_id + datetime (~30 bytes) per row.
    conservative_estimated_bytes = selected_source_bytes + selected_rows * (
        average_user_id_bytes + 32
    )
    print("\n======================================", flush=True)
    print("Pre-extraction exact scan / size estimate", flush=True)
    print("======================================", flush=True)
    print(f"Scanned users: {total_files:,}", flush=True)
    print(f"Selected users: {selected_users:,}", flush=True)
    print(f"Selected rows: {selected_rows:,}", flush=True)
    print(f"Selected source bytes: {format_bytes(selected_source_bytes)}", flush=True)
    print(
        f"Estimated final CSV size (conservative): "
        f"{format_bytes(conservative_estimated_bytes)}",
        flush=True,
    )
    print("No sampling will be applied; continuing with the full extraction.\n", flush=True)

    output_fields = ["user_id", *union_columns, "datetime"]
    canonical_columns = {
        column.strip().casefold(): column for column in union_columns
    }
    completed_stats: Counter[str] = Counter()
    completed_users = 0
    completed_rows = 0
    write_started = time.monotonic()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_handle, \
        output_partial.open("w", encoding="utf-8-sig", newline="") as output_handle, \
        summary_partial.open("w", encoding="utf-8-sig", newline="") as summary_handle:
        manifest_reader = csv.DictReader(manifest_handle)
        output_writer = csv.DictWriter(
            output_handle, fieldnames=output_fields, extrasaction="ignore"
        )
        summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
        output_writer.writeheader()
        summary_writer.writeheader()

        for selected_index, manifest_row in enumerate(manifest_reader, start=1):
            relative_path = manifest_row.pop("relative_path")
            source_path = input_dir / Path(relative_path)
            user_id = manifest_row["user_id"]
            try:
                header, rows = read_full_user_rows(source_path)
                if len(rows) != int(manifest_row["event_count"]):
                    raise ValueError(
                        f"row count changed between passes: scan={manifest_row['event_count']} "
                        f"extract={len(rows)}"
                    )
                for _sort_key, values in rows:
                    source_row = {
                        canonical_columns.get(key.strip().casefold(), key): value
                        for key, value in zip(header, values, strict=True)
                    }
                    source_row["user_id"] = user_id
                    timestamp_value = next(
                        (
                            value
                            for key, value in source_row.items()
                            if key.strip().casefold() == "timestamp"
                        ),
                        "",
                    )
                    source_row["datetime"] = utc_datetime(timestamp_value)
                    output_writer.writerow(source_row)
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                failures.append(
                    {
                        "phase": "extract",
                        "source_file": relative_path,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue

            completed_users += 1
            completed_rows += len(rows)
            summary_writer.writerow(
                {field: manifest_row.get(field, "") for field in SUMMARY_FIELDS}
            )
            has_pay = manifest_row["has_pay"] == "1"
            has_refund = manifest_row["has_refund"] == "1"
            completed_stats["pay_users"] += int(has_pay)
            completed_stats["refund_users"] += int(has_refund)
            completed_stats["pay_and_refund_users"] += int(has_pay and has_refund)
            completed_stats["pay_only_users"] += int(has_pay and not has_refund)
            completed_stats["refund_only_users"] += int(has_refund and not has_pay)
            completed_stats["pay_then_refund_users"] += int(
                manifest_row["has_pay_then_refund"] == "1"
            )
            completed_stats["refund_before_first_pay_users"] += int(
                manifest_row["refund_before_first_pay"] == "1"
            )
            if (
                selected_index == 1
                or selected_index == selected_users
                or (
                    args.progress_every > 0
                    and selected_index % args.progress_every == 0
                )
            ):
                print(
                    progress_line(
                        "Writing selected users",
                        selected_index,
                        selected_users,
                        write_started,
                    )
                    + f" | rows={completed_rows:,}",
                    flush=True,
                )

    os.replace(output_partial, output_path)
    os.replace(summary_partial, summary_path)
    manifest_path.unlink(missing_ok=True)
    write_error_csv(errors_path, failures)

    print("\nExtraction finished. Re-reading final CSV for full validation...", flush=True)
    validation = validate_outputs(
        output_path, summary_path, args.validation_progress_every
    )
    output_bytes = output_path.stat().st_size
    all_validation_passed = all(
        validation[key]
        for key in (
            "user_counts_match",
            "all_users_have_pay_or_refund",
            "all_user_row_counts_match",
            "all_payment_flags_match",
        )
    )
    report = {
        "input": str(input_dir),
        "output": str(output_path),
        "summary": str(summary_path),
        "errors": str(errors_path),
        "source_modified": False,
        "timestamp_unit_counts": dict(timestamp_units),
        "timestamp_conversion": (
            "Magnitude-based Unix conversion; KT4 values in the ms range are divided by 1000 "
            "and rendered as UTC ISO-8601. Original timestamp is preserved unchanged."
        ),
        "sort_strategy": (
            "Files/users are processed in lexical user_id order; every user's rows are sorted "
            "by parsed timestamp. A global in-memory sort was avoided because the output is large."
        ),
        "scan": {
            "total_user_files": total_files,
            "selected_users": selected_users,
            "selected_rows": selected_rows,
            "selected_source_bytes": selected_source_bytes,
            "conservative_estimated_output_bytes": int(conservative_estimated_bytes),
            "failed_files": sum(item["phase"] == "scan" for item in failures),
            **scan_stats,
        },
        "completed": {
            "integrated_users": completed_users,
            "integrated_rows": completed_rows,
            "output_bytes": output_bytes,
            "failed_files_total": len(failures),
            **completed_stats,
        },
        "validation": {**validation, "all_checks_passed": all_validation_passed},
        "union_source_columns": union_columns,
        "final_columns": output_fields,
    }
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n======================================")
    print("EdNet KT4 Payment User Extraction")
    print("======================================")
    print(f"전체 사용자 파일: {total_files:,}")
    print(f"Pay 사용자: {completed_stats['pay_users']:,}")
    print(f"Refund 사용자: {completed_stats['refund_users']:,}")
    print(f"Pay + Refund: {completed_stats['pay_and_refund_users']:,}")
    print(f"Pay Only: {completed_stats['pay_only_users']:,}")
    print(f"Refund Only: {completed_stats['refund_only_users']:,}")
    print(f"Pay -> Refund: {completed_stats['pay_then_refund_users']:,}")
    print(
        f"Refund before first Pay: "
        f"{completed_stats['refund_before_first_pay_users']:,}"
    )
    print("--------------------------------------")
    print(f"통합 사용자: {completed_users:,}")
    print(f"통합 Row: {completed_rows:,}")
    print(f"통합 CSV 크기: {format_bytes(output_bytes)}")
    print(f"검증: {'PASS' if all_validation_passed else 'FAIL'}")
    print(f"결과: {output_path}")
    print(f"요약: {summary_path}")
    print(f"오류: {errors_path} ({len(failures):,}건)")
    print(f"보고서: {report_path}")
    return 0 if all_validation_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. KT4 source files were not modified.", file=sys.stderr)
        raise SystemExit(130)
