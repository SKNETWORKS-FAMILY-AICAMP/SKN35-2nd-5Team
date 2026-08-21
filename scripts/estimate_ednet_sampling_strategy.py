#!/usr/bin/env python3
"""Compare EdNet 7/14/30-day windows and user-level sampling strategies.

This script creates only small EDA estimate artifacts. It never writes an
integrated event-log CSV, never samples individual rows, and never creates a
churn label.

Definitions
-----------
* Pay + Refund window: [first_refund - N days, first_refund], inclusive.
* Pay Only window: [last_pay, last_pay + N days], inclusive.
* Quality events exclude transaction actions (pay/refund).
* Pay + Refund quality uses all learning events before first_refund.
* Pay Only quality uses all learning events after last_pay.
* observation_days measures anchor coverage: earliest learning event to refund
  for Pay + Refund; last_pay to latest later learning event for Pay Only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "KT4"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "ednet_payment_user_summary.csv.manifest.tmp"
)
DEFAULT_CSV = PROJECT_ROOT / "artifacts" / "eda" / "ednet_sampling_estimate.csv"
DEFAULT_JSON = PROJECT_ROOT / "artifacts" / "eda" / "ednet_sampling_estimate.json"
WINDOW_DAYS = (7, 14, 30)
RATIOS = (1, 2, 3, 5)
SECONDS_PER_DAY = 86_400.0
TRANSACTION_ACTIONS = {"pay", "refund"}
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate EdNet window and Pay Only user-sampling candidates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
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
    raise ValueError(f"unselected manifest user: {row['user_id']}")


def event_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 9:
        return "1-9"
    if value <= 49:
        return "10-49"
    if value <= 99:
        return "50-99"
    if value <= 499:
        return "100-499"
    return "500+"


def active_day_bin(value: int) -> str:
    if value == 0:
        return "0 days"
    if value == 1:
        return "1 day"
    if value <= 3:
        return "2-3 days"
    if value <= 7:
        return "4-7 days"
    if value <= 14:
        return "8-14 days"
    return "15+ days"


def first_pay_quarter(raw: str) -> str:
    seconds, _unit = parse_timestamp(raw)
    if seconds is None:
        return "unknown"
    value = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{value.year}-Q{(value.month - 1) // 3 + 1}"


def platform_group(counts: Counter[str]) -> str:
    meaningful = Counter(
        {key: value for key, value in counts.items() if key not in {"", "<missing>"}}
    )
    total = sum(meaningful.values())
    if not total:
        return "unknown"
    most_common, count = meaningful.most_common(1)[0]
    if count / total >= 0.8:
        return most_common
    return "mixed"


def describe(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def distribution(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record[field]) for record in records)
    return dict(sorted(counts.items()))


def total_variation_distance(
    population: Sequence[dict[str, Any]], sample: Sequence[dict[str, Any]], field: str
) -> float:
    population_counts = Counter(str(record[field]) for record in population)
    sample_counts = Counter(str(record[field]) for record in sample)
    categories = set(population_counts) | set(sample_counts)
    return 0.5 * sum(
        abs(
            population_counts[category] / len(population)
            - sample_counts[category] / len(sample)
        )
        for category in categories
    )


def stable_tie_value(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_sample(
    population: Sequence[dict[str, Any]], target: int, seed: int
) -> list[dict[str, Any]]:
    """Exact-size proportional sample, with random user selection inside strata."""
    if target >= len(population):
        return list(population)
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in population:
        key = (
            record["first_pay_quarter"],
            record["quality_event_bin"],
            record["quality_active_day_bin"],
            record["platform_group"],
        )
        strata[key].append(record)

    rng = random.Random(seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])

    total = len(population)
    allocations: dict[tuple[str, ...], int] = {}
    fractions: list[tuple[float, int, tuple[str, ...]]] = []
    allocated = 0
    for key, records in strata.items():
        ideal = len(records) * target / total
        base = min(len(records), math.floor(ideal))
        allocations[key] = base
        allocated += base
        fractions.append(
            (ideal - base, stable_tie_value(seed, "|".join(key)), key)
        )

    for _fraction, _tie, key in sorted(fractions, reverse=True):
        if allocated >= target:
            break
        if allocations[key] < len(strata[key]):
            allocations[key] += 1
            allocated += 1
    if allocated < target:
        for key in sorted(strata, key=lambda value: stable_tie_value(seed, str(value))):
            while allocations[key] < len(strata[key]) and allocated < target:
                allocations[key] += 1
                allocated += 1
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        selected.extend(strata[key][: allocations[key]])
    if len(selected) != target:
        raise RuntimeError(f"sampling allocation error: expected {target}, got {len(selected)}")
    return selected


def format_bytes(value: int | float) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.3f} GiB"
    return f"{value / 1024**2:.3f} MiB"


def analyze_user(
    source_path: Path,
    manifest: dict[str, str],
    buffer: io.StringIO,
    size_writer: csv.writer,
) -> tuple[dict[str, Any], tuple[str, ...], list[str], Counter[str]]:
    group = group_for_manifest(manifest)
    if group == "refund_only":
        raise ValueError("refund_only is not a learning-candidate group")
    anchor_raw = (
        manifest["first_refund_timestamp"]
        if group == "pay_and_refund"
        else manifest["last_pay_timestamp"]
    )
    anchor, _unit = parse_timestamp(anchor_raw)
    if anchor is None:
        raise ValueError("missing valid anchor timestamp")

    first_pay, _unit = parse_timestamp(manifest["first_pay_timestamp"])
    rows_by_window = {days: 0 for days in WINDOW_DAYS}
    bytes_by_window = {days: 0 for days in WINDOW_DAYS}
    learning_times: list[float] = []
    active_dates: set[str] = set()
    platform_counts: Counter[str] = Counter()
    timestamp_units: Counter[str] = Counter()
    anchor_found = False
    last_pay_item_id = "<missing>"

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        normalized = tuple(column.strip().casefold() for column in header)
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate columns after case normalization")
        lookup = {name: index for index, name in enumerate(normalized)}
        if not {"timestamp", "action_type"}.issubset(lookup):
            raise ValueError("missing timestamp or action_type")
        timestamp_index = lookup["timestamp"]
        action_index = lookup["action_type"]
        platform_index = lookup.get("platform")
        item_index = lookup.get("item_id")
        required_index = max(timestamp_index, action_index)

        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) <= required_index:
                raise ValueError(f"row {row_number} missing required fields")
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            elif len(row) > len(header):
                raise ValueError(
                    f"row {row_number} has {len(row)} values; header has {len(header)}"
                )
            seconds, unit = parse_timestamp(row[timestamp_index])
            timestamp_units[unit] += 1
            if seconds is None:
                continue
            action = row[action_index].strip().casefold()
            if platform_index is not None:
                platform_counts[row[platform_index].strip().casefold() or "<missing>"] += 1
            if action == "pay" and seconds == anchor and item_index is not None:
                last_pay_item_id = row[item_index].strip() or "<missing>"

            if group == "pay_and_refund":
                distance_days = (anchor - seconds) / SECONDS_PER_DAY
                in_direction = seconds <= anchor
                anchor_action = "refund"
                is_quality = seconds < anchor and action not in TRANSACTION_ACTIONS
            else:
                distance_days = (seconds - anchor) / SECONDS_PER_DAY
                in_direction = seconds >= anchor
                anchor_action = "pay"
                is_quality = seconds > anchor and action not in TRANSACTION_ACTIONS

            if action == anchor_action and seconds == anchor:
                anchor_found = True
            if is_quality:
                learning_times.append(seconds)
                active_dates.add(
                    datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
                )
            if in_direction and distance_days >= 0:
                output_values = [manifest["user_id"], *row, iso_utc(seconds)]
                row_bytes = serialized_size(size_writer, buffer, output_values)
                for days in WINDOW_DAYS:
                    if distance_days <= days:
                        rows_by_window[days] += 1
                        bytes_by_window[days] += row_bytes

    if learning_times:
        observation_days = (
            (anchor - min(learning_times)) / SECONDS_PER_DAY
            if group == "pay_and_refund"
            else (max(learning_times) - anchor) / SECONDS_PER_DAY
        )
    else:
        observation_days = 0.0
    active_days = len(active_dates)
    event_count = len(learning_times)
    record: dict[str, Any] = {
        "user_id": manifest["user_id"],
        "group": group,
        "quality_event_count": event_count,
        "quality_active_days": active_days,
        "quality_observation_days": observation_days,
        "quality_event_bin": event_bin(event_count),
        "quality_active_day_bin": active_day_bin(active_days),
        "first_pay_quarter": first_pay_quarter(manifest["first_pay_timestamp"]),
        "first_pay_seconds": first_pay,
        "platform_group": platform_group(platform_counts),
        "last_pay_item_id": last_pay_item_id,
        "anchor_event_retained": anchor_found,
    }
    for days in WINDOW_DAYS:
        record[f"rows_{days}d"] = rows_by_window[days]
        record[f"bytes_{days}d"] = bytes_by_window[days]
    return record, normalized, header, timestamp_units


def quality_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "users": len(records),
        "event_count": describe([record["quality_event_count"] for record in records]),
        "active_days": describe([record["quality_active_days"] for record in records]),
        "observation_days": describe(
            [record["quality_observation_days"] for record in records]
        ),
        "event_count_bins": {
            label: sum(record["quality_event_bin"] == label for record in records)
            for label in ("0", "1-9", "10-49", "50-99", "100-499", "500+")
        },
        "active_day_bins": {
            label: sum(record["quality_active_day_bin"] == label for record in records)
            for label in (
                "0 days",
                "1 day",
                "2-3 days",
                "4-7 days",
                "8-14 days",
                "15+ days",
            )
        },
        "observation_under_7_days": sum(
            record["quality_observation_days"] < 7 for record in records
        ),
        "fewer_than_10_events": sum(
            record["quality_event_count"] < 10 for record in records
        ),
        "anchor_event_retained_users": sum(
            record["anchor_event_retained"] for record in records
        ),
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"KT4 directory not found: {input_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"selection manifest not found: {manifest_path}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_manifest = list(csv.DictReader(handle))
    manifest_rows = [
        row for row in all_manifest if group_for_manifest(row) != "refund_only"
    ]
    refund_only_users = len(all_manifest) - len(manifest_rows)
    print(
        f"Learning candidates: {len(manifest_rows):,} "
        f"(refund-only excluded: {refund_only_users:,})",
        flush=True,
    )

    buffer = io.StringIO(newline="")
    size_writer = csv.writer(buffer)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    timestamp_units: Counter[str] = Counter()
    schemas: dict[tuple[str, ...], list[str]] = {}
    started = time.monotonic()
    for index, manifest in enumerate(manifest_rows, start=1):
        source_path = input_dir / Path(manifest["relative_path"])
        try:
            record, schema, header, file_units = analyze_user(
                source_path, manifest, buffer, size_writer
            )
        except (OSError, UnicodeError, csv.Error, ValueError, StopIteration) as exc:
            failures.append(
                {
                    "user_id": manifest["user_id"],
                    "source_file": manifest["relative_path"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        else:
            records.append(record)
            timestamp_units.update(file_units)
            schemas.setdefault(schema, header)
        if (
            index == 1
            or index == len(manifest_rows)
            or (args.progress_every > 0 and index % args.progress_every == 0)
        ):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            remaining = (len(manifest_rows) - index) / rate if rate else float("inf")
            print(
                f"Analyzing users: {index:,} / {len(manifest_rows):,} | "
                f"{rate:.1f} users/s | ETA {int(remaining // 60):02d}:{int(remaining % 60):02d}",
                flush=True,
            )

    positive_records = [record for record in records if record["group"] == "pay_and_refund"]
    pay_only_records = [record for record in records if record["group"] == "pay_only"]
    if not positive_records or not pay_only_records:
        raise RuntimeError("both Pay + Refund and Pay Only records are required")

    # All observed KT4 files currently share one schema. Include one final CSV header.
    union_columns: list[str] = []
    seen: set[str] = set()
    for schema, header in schemas.items():
        for normalized, original in zip(schema, header, strict=True):
            if normalized in {"user_id", "datetime"}:
                continue
            if normalized not in seen:
                seen.add(normalized)
                union_columns.append(original)
    final_header = ["user_id", *union_columns, "datetime"]
    header_bytes = serialized_size(size_writer, buffer, final_header)

    full_window_stats: dict[str, Any] = {}
    for days in WINDOW_DAYS:
        positive_rows = sum(record[f"rows_{days}d"] for record in positive_records)
        positive_bytes = sum(record[f"bytes_{days}d"] for record in positive_records)
        pay_only_rows = sum(record[f"rows_{days}d"] for record in pay_only_records)
        pay_only_bytes = sum(record[f"bytes_{days}d"] for record in pay_only_records)
        full_window_stats[str(days)] = {
            "pay_and_refund": {
                "users": len(positive_records),
                "rows": positive_rows,
                "mean_rows_per_user": positive_rows / len(positive_records),
                "estimated_bytes": positive_bytes + header_bytes,
                "estimated_size": format_bytes(positive_bytes + header_bytes),
            },
            "pay_only": {
                "users": len(pay_only_records),
                "rows": pay_only_rows,
                "mean_rows_per_user": pay_only_rows / len(pay_only_records),
                "estimated_bytes": pay_only_bytes + header_bytes,
                "estimated_size": format_bytes(pay_only_bytes + header_bytes),
            },
            "total": {
                "users": len(records),
                "rows": positive_rows + pay_only_rows,
                "estimated_bytes": positive_bytes + pay_only_bytes + header_bytes,
                "estimated_size": format_bytes(
                    positive_bytes + pay_only_bytes + header_bytes
                ),
            },
        }

    samples: dict[int, list[dict[str, Any]]] = {}
    sampling_diagnostics: dict[str, Any] = {}
    for ratio in RATIOS:
        target = min(len(positive_records) * ratio, len(pay_only_records))
        sample = stratified_sample(pay_only_records, target, RANDOM_SEED)
        samples[ratio] = sample
        sampling_diagnostics[f"1:{ratio}"] = {
            "target_users": target,
            "seed": RANDOM_SEED,
            "stratification_fields": [
                "first_pay_quarter",
                "quality_event_bin",
                "quality_active_day_bin",
                "platform_group",
            ],
            "total_variation_distance": {
                field: total_variation_distance(pay_only_records, sample, field)
                for field in (
                    "first_pay_quarter",
                    "quality_event_bin",
                    "quality_active_day_bin",
                    "platform_group",
                    "last_pay_item_id",
                )
            },
            "sample_distributions": {
                field: distribution(sample, field)
                for field in (
                    "first_pay_quarter",
                    "quality_event_bin",
                    "quality_active_day_bin",
                    "platform_group",
                    "last_pay_item_id",
                )
            },
        }

    estimate_rows: list[dict[str, Any]] = []
    for days in WINDOW_DAYS:
        positive_rows = sum(record[f"rows_{days}d"] for record in positive_records)
        positive_bytes = sum(record[f"bytes_{days}d"] for record in positive_records)
        for ratio in RATIOS:
            sample = samples[ratio]
            pay_rows = sum(record[f"rows_{days}d"] for record in sample)
            pay_bytes = sum(record[f"bytes_{days}d"] for record in sample)
            total_rows = positive_rows + pay_rows
            total_bytes = positive_bytes + pay_bytes + header_bytes
            estimate_rows.append(
                {
                    "window_days": days,
                    "ratio": f"1:{ratio}",
                    "refund_users": len(positive_records),
                    "pay_only_users": len(sample),
                    "total_users": len(positive_records) + len(sample),
                    "refund_rows": positive_rows,
                    "pay_only_rows": pay_rows,
                    "estimated_rows": total_rows,
                    "estimated_bytes": total_bytes,
                    "estimated_mib": total_bytes / 1024**2,
                    "estimated_gib": total_bytes / 1024**3,
                    "positive_share": len(positive_records)
                    / (len(positive_records) + len(sample)),
                    "class_ratio": f"1:{len(sample) / len(positive_records):.3f}",
                }
            )

    csv_fields = [
        "window_days",
        "ratio",
        "refund_users",
        "pay_only_users",
        "total_users",
        "refund_rows",
        "pay_only_rows",
        "estimated_rows",
        "estimated_bytes",
        "estimated_mib",
        "estimated_gib",
        "positive_share",
        "class_ratio",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(estimate_rows)

    product_counts = Counter(record["last_pay_item_id"] for record in pay_only_records)
    report = {
        "creates_integrated_csv": False,
        "random_seed": RANDOM_SEED,
        "window_definitions": {
            "pay_and_refund": "inclusive [first_refund - N days, first_refund]",
            "pay_only": "inclusive [last_pay, last_pay + N days]",
            "quality_events": "all non-pay/non-refund events on the anchor side",
            "observation_days": (
                "anchor coverage: refund - earliest prior learning event, or latest later "
                "learning event - last_pay"
            ),
        },
        "anchor_design_warning": (
            "The two classes use opposite temporal directions and different lifecycle anchors. "
            "Pay Only windows after last_pay can encode follow-up availability/right censoring, "
            "while refund windows end at a known future outcome. A later label design should "
            "align an index time and observation direction before modeling."
        ),
        "sampling_method": {
            "unit": "user_id",
            "method": "proportional composite-stratified sampling",
            "fields": [
                "first_pay_quarter",
                "quality_event_bin",
                "quality_active_day_bin",
                "platform_group",
            ],
            "product_type_decision": (
                "KT4 pay item_id is available but undocumented/high-cardinality; it was not "
                "treated as a semantic product type. Its distribution is measured after sampling."
            ),
        },
        "pay_and_refund_quality": quality_report(positive_records),
        "pay_only_quality": quality_report(pay_only_records),
        "pay_only_population_distributions": {
            field: distribution(pay_only_records, field)
            for field in (
                "first_pay_quarter",
                "quality_event_bin",
                "quality_active_day_bin",
                "platform_group",
                "last_pay_item_id",
            )
        },
        "pay_product_metadata": {
            "unique_last_pay_item_ids": len(product_counts),
            "missing_last_pay_item_id_users": product_counts["<missing>"],
            "top_20_last_pay_item_ids": product_counts.most_common(20),
        },
        "full_population_windows": full_window_stats,
        "sampling_estimates": estimate_rows,
        "sampling_diagnostics": sampling_diagnostics,
        "refund_only_excluded_users": refund_only_users,
        "timestamp_unit_counts": dict(timestamp_units),
        "source_schema_count": len(schemas),
        "final_columns_assumed_for_size": final_header,
        "failures": failures,
        "artifacts": {
            "csv": str(output_csv),
            "json": str(output_json),
        },
    }
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n======================================")
    print("EdNet Sampling Strategy Estimate")
    print("======================================")
    print(
        f"Pay + Refund quality: users={len(positive_records):,}, "
        f"mean events={report['pay_and_refund_quality']['event_count']['mean']:.2f}, "
        f"median events={report['pay_and_refund_quality']['event_count']['median']:.2f}, "
        f"mean active_days={report['pay_and_refund_quality']['active_days']['mean']:.2f}, "
        f"median active_days={report['pay_and_refund_quality']['active_days']['median']:.2f}"
    )
    print(
        f"Pay Only quality: users={len(pay_only_records):,}, "
        f"mean events={report['pay_only_quality']['event_count']['mean']:.2f}, "
        f"median events={report['pay_only_quality']['event_count']['median']:.2f}, "
        f"mean active_days={report['pay_only_quality']['active_days']['mean']:.2f}, "
        f"median active_days={report['pay_only_quality']['active_days']['median']:.2f}"
    )
    for days in WINDOW_DAYS:
        stats = full_window_stats[str(days)]
        print(
            f"{days}d full: refund={stats['pay_and_refund']['rows']:,} rows, "
            f"pay_only={stats['pay_only']['rows']:,} rows, "
            f"total={stats['total']['rows']:,} rows / "
            f"{stats['total']['estimated_size']}"
        )
    print(f"Failures: {len(failures):,}")
    print(f"CSV estimate: {output_csv}")
    print(f"JSON report: {output_json}")
    print("No integrated raw-log CSV was created.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
