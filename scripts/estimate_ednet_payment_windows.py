#!/usr/bin/env python3
"""Estimate bounded EdNet payment/refund windows without creating final CSVs.

Window definitions (inclusive so the anchor transaction is retained):

* pay + refund: [first_refund - 30 days, first_refund]
* refund only: [first_refund - 30 days, first_refund], reported separately
* pay only: [last_pay, last_pay + 30 days]

The script reuses the completed selection manifest from the extraction scan and
streams each selected source file once.  It counts rows and simulates CSV byte
serialization in memory, but does not create an integrated data CSV.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "KT4"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "ednet_payment_user_summary.csv.manifest.tmp"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "artifacts" / "eda" / "ednet_payment_window_estimate.json"
)
WINDOW_SECONDS = 30 * 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count and size 30-day KT4 transaction-centered windows only."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--progress-every", type=int, default=1_000)
    return parser.parse_args()


def parse_timestamp(raw: str) -> tuple[float | None, str]:
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


def iso_utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


def serialized_size(writer: csv.writer, buffer: io.StringIO, values: list[str]) -> int:
    buffer.seek(0)
    buffer.truncate(0)
    writer.writerow(values)
    return len(buffer.getvalue().encode("utf-8"))


def group_for_manifest(row: dict[str, str]) -> str:
    has_pay = row["has_pay"] == "1"
    has_refund = row["has_refund"] == "1"
    if has_pay and has_refund:
        return "pay_and_refund"
    if has_pay:
        return "pay_only"
    if has_refund:
        return "refund_only"
    raise ValueError(f"manifest contains unselected user: {row['user_id']}")


def format_bytes(size: int | float) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.3f} GiB"
    return f"{size / 1024**2:.3f} MiB"


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"KT4 directory not found: {input_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Completed selection manifest not found: {manifest_path}. "
            "Run the selection scan first."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    total_users = len(manifest_rows)
    print(f"Selected manifest users: {total_users:,}", flush=True)

    group_users: Counter[str] = Counter()
    group_rows: Counter[str] = Counter()
    group_bytes: Counter[str] = Counter()
    group_source_lifetime_rows: Counter[str] = Counter()
    group_anchor_found: Counter[str] = Counter()
    timestamp_units: Counter[str] = Counter()
    invalid_timestamp_rows: Counter[str] = Counter()
    schemas: dict[tuple[str, ...], list[str]] = {}
    rows_by_group_schema: Counter[tuple[str, tuple[str, ...]]] = Counter()
    failures: list[dict[str, str]] = []
    no_valid_anchor: list[str] = []

    buffer = io.StringIO(newline="")
    size_writer = csv.writer(buffer)
    started = time.monotonic()
    for index, manifest in enumerate(manifest_rows, start=1):
        group = group_for_manifest(manifest)
        group_users[group] += 1
        anchor_raw = (
            manifest["first_refund_timestamp"]
            if group in {"pay_and_refund", "refund_only"}
            else manifest["last_pay_timestamp"]
        )
        anchor, _anchor_unit = parse_timestamp(anchor_raw)
        if anchor is None:
            no_valid_anchor.append(manifest["user_id"])
            continue
        if group in {"pay_and_refund", "refund_only"}:
            lower_bound, upper_bound = anchor - WINDOW_SECONDS, anchor
            anchor_action = "refund"
        else:
            lower_bound, upper_bound = anchor, anchor + WINDOW_SECONDS
            anchor_action = "pay"

        source_path = input_dir / Path(manifest["relative_path"])
        try:
            with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader)
                normalized = tuple(column.strip().casefold() for column in header)
                if len(set(normalized)) != len(normalized):
                    raise ValueError("duplicate columns after case normalization")
                lookup = {name: idx for idx, name in enumerate(normalized)}
                if not {"timestamp", "action_type"}.issubset(lookup):
                    raise ValueError("missing timestamp or action_type")
                timestamp_index = lookup["timestamp"]
                action_index = lookup["action_type"]
                required_index = max(timestamp_index, action_index)
                schemas.setdefault(normalized, header)

                source_rows = 0
                selected_rows = 0
                found_anchor = False
                for row_number, row in enumerate(reader, start=2):
                    if not row or all(not value.strip() for value in row):
                        continue
                    if len(row) <= required_index:
                        raise ValueError(f"row {row_number} is missing required fields")
                    if len(row) < len(header):
                        row.extend([""] * (len(header) - len(row)))
                    elif len(row) > len(header):
                        raise ValueError(
                            f"row {row_number} has {len(row)} values; header has {len(header)}"
                        )
                    source_rows += 1
                    seconds, unit = parse_timestamp(row[timestamp_index])
                    timestamp_units[unit] += 1
                    if seconds is None:
                        invalid_timestamp_rows[group] += 1
                        continue
                    if lower_bound <= seconds <= upper_bound:
                        selected_rows += 1
                        action = row[action_index].strip().casefold()
                        if action == anchor_action and seconds == anchor:
                            found_anchor = True
                        output_values = [
                            manifest["user_id"],
                            *row,
                            iso_utc(seconds),
                        ]
                        group_bytes[group] += serialized_size(
                            size_writer, buffer, output_values
                        )
                group_source_lifetime_rows[group] += source_rows
                group_rows[group] += selected_rows
                rows_by_group_schema[(group, normalized)] += selected_rows
                group_anchor_found[group] += int(found_anchor)
        except (OSError, UnicodeError, csv.Error, ValueError, StopIteration) as exc:
            failures.append(
                {
                    "user_id": manifest["user_id"],
                    "source_file": manifest["relative_path"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

        if (
            index == 1
            or index == total_users
            or (args.progress_every > 0 and index % args.progress_every == 0)
        ):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            remaining = (total_users - index) / rate if rate else float("inf")
            print(
                f"Estimating windows: {index:,} / {total_users:,} | "
                f"rows={sum(group_rows.values()):,} | "
                f"{rate:.1f} users/s | ETA {int(remaining // 60):02d}:{int(remaining % 60):02d}",
                flush=True,
            )

    # If schemas differ, final union output adds one empty CSV field (one comma)
    # per missing column per row. Include this small adjustment in the estimate.
    union_normalized: list[str] = []
    union_original: list[str] = []
    seen: set[str] = set()
    for schema, header in schemas.items():
        for normalized_name, original_name in zip(schema, header, strict=True):
            if normalized_name in {"user_id", "datetime"}:
                continue
            if normalized_name not in seen:
                seen.add(normalized_name)
                union_normalized.append(normalized_name)
                union_original.append(original_name)
    for (group, schema), row_count in rows_by_group_schema.items():
        source_columns = {
            name for name in schema if name not in {"user_id", "datetime"}
        }
        missing_columns = len(set(union_normalized) - source_columns)
        group_bytes[group] += row_count * missing_columns

    final_header = ["user_id", *union_original, "datetime"]
    header_bytes = serialized_size(size_writer, buffer, final_header)
    for group in group_users:
        group_bytes[group] += header_bytes

    candidate_groups = ("pay_and_refund", "pay_only")
    candidate_users = sum(group_users[group] for group in candidate_groups)
    candidate_rows = sum(group_rows[group] for group in candidate_groups)
    # Candidate groups would share one output header rather than one per group.
    candidate_bytes = (
        sum(group_bytes[group] for group in candidate_groups) - header_bytes
    )
    refund_only_users = group_users["refund_only"]
    refund_only_rows = group_rows["refund_only"]
    refund_only_bytes = group_bytes["refund_only"]
    selected_lifetime_rows = sum(group_source_lifetime_rows.values())
    bounded_rows = candidate_rows + refund_only_rows

    result_groups: dict[str, dict[str, Any]] = {}
    for group in ("pay_and_refund", "pay_only", "refund_only"):
        lifetime = group_source_lifetime_rows[group]
        window_rows = group_rows[group]
        result_groups[group] = {
            "users": group_users[group],
            "window_rows": window_rows,
            "estimated_csv_bytes": group_bytes[group],
            "estimated_csv_size": format_bytes(group_bytes[group]),
            "source_lifetime_rows": lifetime,
            "retained_row_ratio": window_rows / lifetime if lifetime else None,
            "users_with_anchor_event_retained": group_anchor_found[group],
            "invalid_timestamp_rows_excluded": invalid_timestamp_rows[group],
        }

    report: dict[str, Any] = {
        "creates_integrated_csv": False,
        "window_days": 30,
        "boundaries_are_inclusive": True,
        "definitions": {
            "pay_and_refund": "first_refund - 30 days through first_refund",
            "pay_only": "last_pay through last_pay + 30 days",
            "refund_only": (
                "first_refund - 30 days through first_refund; kept separate and excluded "
                "from the learning-candidate totals"
            ),
        },
        "groups": result_groups,
        "learning_candidates": {
            "included_groups": list(candidate_groups),
            "users": candidate_users,
            "rows": candidate_rows,
            "estimated_csv_bytes": candidate_bytes,
            "estimated_csv_size": format_bytes(candidate_bytes),
        },
        "refund_only_separate": {
            "users": refund_only_users,
            "rows": refund_only_rows,
            "estimated_csv_bytes": refund_only_bytes,
            "estimated_csv_size": format_bytes(refund_only_bytes),
        },
        "all_bounded_outputs": {
            "users": candidate_users + refund_only_users,
            "rows": bounded_rows,
            "estimated_csv_bytes": candidate_bytes + refund_only_bytes,
            "estimated_csv_size": format_bytes(candidate_bytes + refund_only_bytes),
            "selected_lifetime_rows": selected_lifetime_rows,
            "retained_row_ratio": (
                bounded_rows / selected_lifetime_rows if selected_lifetime_rows else None
            ),
        },
        "pay_refund_sequence_context": {
            "pay_then_refund_users": sum(
                row["has_pay_then_refund"] == "1" for row in manifest_rows
            ),
            "refund_before_first_pay_users": sum(
                row["refund_before_first_pay"] == "1" for row in manifest_rows
            ),
        },
        "timestamp_unit_counts": dict(timestamp_units),
        "source_schemas": [list(schema) for schema in schemas],
        "union_output_columns": final_header,
        "users_without_valid_anchor": no_valid_anchor,
        "failures": failures,
    }
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n======================================")
    print("EdNet 30-day Window Size Estimate")
    print("======================================")
    print(
        f"Pay + Refund: {group_users['pay_and_refund']:,} users / "
        f"{group_rows['pay_and_refund']:,} rows / {format_bytes(group_bytes['pay_and_refund'])}"
    )
    print(
        f"Pay Only: {group_users['pay_only']:,} users / "
        f"{group_rows['pay_only']:,} rows / {format_bytes(group_bytes['pay_only'])}"
    )
    print(
        f"Learning candidates: {candidate_users:,} users / {candidate_rows:,} rows / "
        f"{format_bytes(candidate_bytes)}"
    )
    print("--------------------------------------")
    print(
        f"Refund Only (separate): {refund_only_users:,} users / "
        f"{refund_only_rows:,} rows / {format_bytes(refund_only_bytes)}"
    )
    print("--------------------------------------")
    print(
        f"All bounded outputs: {candidate_users + refund_only_users:,} users / "
        f"{bounded_rows:,} rows / {format_bytes(candidate_bytes + refund_only_bytes)}"
    )
    print(
        f"Rows retained vs selected lifetime: {bounded_rows:,} / "
        f"{selected_lifetime_rows:,} ({bounded_rows / selected_lifetime_rows:.2%})"
    )
    print(f"Failures: {len(failures):,}")
    print(f"Estimate report: {report_path}")
    print("No integrated CSV was created.")
    return 0 if not failures and not no_valid_anchor else 2


if __name__ == "__main__":
    raise SystemExit(main())
