#!/usr/bin/env python3
"""Analyze common-pay-anchor observation/outcome windows for EdNet KT4.

This script performs statistics only. It does not write event logs, feature
datasets, labels, samples, or model artifacts.

Primary analysis unit
---------------------
One sample candidate per user, anchored at first_pay:

    pay -> feature observation -> prediction point -> outcome window

Classification for each observation/outcome combination:

* refund inside feature observation: early outcome, excluded;
* feature window beyond global data end: observation incomplete, excluded;
* refund after prediction point and within outcome window: eligible positive;
* no such refund and the full outcome window ends by global data end:
  eligible non-refund candidate;
* otherwise: right-censored, excluded.

The same rules are also counted for every payment episode as a diagnostic, but
episode results are explicitly not independent training samples: windows may
overlap and one refund may be visible from multiple payment anchors because KT4
does not provide a transaction/refund linkage key.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "KT4"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "ednet_payment_user_summary.csv.manifest.tmp"
)
DEFAULT_CSV = (
    PROJECT_ROOT / "artifacts" / "eda" / "ednet_pay_anchor_window_analysis.csv"
)
DEFAULT_JSON = (
    PROJECT_ROOT / "artifacts" / "eda" / "ednet_pay_anchor_window_analysis.json"
)
COMBINATIONS = ((7, 14), (14, 30), (14, 60), (30, 30))
OBSERVATION_DAYS = tuple(sorted({observation for observation, _outcome in COMBINATIONS}))
SECONDS_PER_DAY = 86_400.0
TRANSACTION_ACTIONS = {"pay", "refund"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pay-anchored EdNet observation/outcome windows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--progress-every", type=int, default=10_000)
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


def iso_utc(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


def serialized_size(writer: csv.writer, buffer: io.StringIO, values: list[str]) -> int:
    buffer.seek(0)
    buffer.truncate(0)
    writer.writerow(values)
    return len(buffer.getvalue().encode("utf-8"))


def describe(values: Sequence[int | float]) -> dict[str, int | float | None]:
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


def format_bytes(value: int | float) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.3f} GiB"
    return f"{value / 1024**2:.3f} MiB"


def analyze_selected_file(
    path: Path,
    manifest: dict[str, str],
    buffer: io.StringIO,
    size_writer: csv.writer,
) -> tuple[dict[str, Any], float | None, Counter[str], tuple[str, ...], list[str]]:
    first_pay, _unit = parse_timestamp(manifest["first_pay_timestamp"])
    if first_pay is None:
        raise ValueError("selected pay user has no valid first_pay_timestamp")

    pay_times: list[float] = []
    refund_times: list[float] = []
    metrics = {
        days: {
            "learning_event_count": 0,
            "active_dates": set(),
            "raw_rows": 0,
            "raw_bytes": 0,
        }
        for days in OBSERVATION_DAYS
    }
    timestamp_units: Counter[str] = Counter()
    file_max: float | None = None
    previous_timestamp: float | None = None
    monotonic = True

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        schema = tuple(column.strip().casefold() for column in header)
        if len(set(schema)) != len(schema):
            raise ValueError("duplicate columns after case normalization")
        lookup = {name: index for index, name in enumerate(schema)}
        if not {"timestamp", "action_type"}.issubset(lookup):
            raise ValueError("missing timestamp or action_type")
        timestamp_index = lookup["timestamp"]
        action_index = lookup["action_type"]
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
            if previous_timestamp is not None and seconds < previous_timestamp:
                monotonic = False
            previous_timestamp = seconds
            file_max = seconds if file_max is None else max(file_max, seconds)
            action = row[action_index].strip().casefold()
            if action == "pay":
                pay_times.append(seconds)
            elif action == "refund":
                refund_times.append(seconds)

            elapsed_days = (seconds - first_pay) / SECONDS_PER_DAY
            if elapsed_days < 0:
                continue
            relevant_windows = [days for days in OBSERVATION_DAYS if elapsed_days < days]
            if not relevant_windows:
                continue
            row_bytes = serialized_size(
                size_writer,
                buffer,
                [manifest["user_id"], *row, iso_utc(seconds)],
            )
            for days in relevant_windows:
                metric = metrics[days]
                metric["raw_rows"] += 1
                metric["raw_bytes"] += row_bytes
                if action not in TRANSACTION_ACTIONS:
                    metric["learning_event_count"] += 1
                    metric["active_dates"].add(
                        datetime.fromtimestamp(seconds, tz=timezone.utc)
                        .date()
                        .isoformat()
                    )

    pay_times.sort()
    refund_times.sort()
    if len(pay_times) != int(manifest["pay_count"]):
        raise ValueError(
            f"pay_count mismatch: manifest={manifest['pay_count']}, parsed={len(pay_times)}"
        )
    record = {
        "user_id": manifest["user_id"],
        "first_pay": first_pay,
        "pay_times": pay_times,
        "refund_times": refund_times,
        "file_timestamp_monotonic": monotonic,
        "metrics": {
            days: {
                "learning_event_count": metric["learning_event_count"],
                "active_days": len(metric["active_dates"]),
                "raw_rows": metric["raw_rows"],
                "raw_bytes": metric["raw_bytes"],
            }
            for days, metric in metrics.items()
        },
    }
    return record, file_max, timestamp_units, schema, header


def scan_unselected_file(
    path: Path,
) -> tuple[float | None, Counter[str], bool]:
    timestamp_units: Counter[str] = Counter()
    file_max: float | None = None
    previous_timestamp: float | None = None
    monotonic = True
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        schema = tuple(column.strip().casefold() for column in header)
        if "timestamp" not in schema:
            raise ValueError("missing timestamp")
        timestamp_index = schema.index("timestamp")
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) <= timestamp_index:
                raise ValueError(f"row {row_number} missing timestamp")
            seconds, unit = parse_timestamp(row[timestamp_index])
            timestamp_units[unit] += 1
            if seconds is None:
                continue
            if previous_timestamp is not None and seconds < previous_timestamp:
                monotonic = False
            previous_timestamp = seconds
            file_max = seconds if file_max is None else max(file_max, seconds)
    return file_max, timestamp_units, monotonic


def refunds_in_interval(
    refund_times: Sequence[float], start: float, end: float, end_inclusive: bool
) -> list[float]:
    if end_inclusive:
        return [value for value in refund_times if start <= value <= end]
    return [value for value in refund_times if start <= value < end]


def classify_anchor(
    anchor: float,
    refund_times: Sequence[float],
    observation_days: int,
    outcome_days: int,
    global_end: float,
) -> str:
    prediction = anchor + observation_days * SECONDS_PER_DAY
    outcome_end = prediction + outcome_days * SECONDS_PER_DAY
    if refunds_in_interval(refund_times, anchor, prediction, end_inclusive=False):
        return "refund_during_feature"
    if prediction > global_end:
        return "observation_incomplete"
    if refunds_in_interval(refund_times, prediction, outcome_end, end_inclusive=True):
        return "positive"
    if outcome_end <= global_end:
        return "non_refund"
    return "right_censored"


def build_combination_result(
    records: Sequence[dict[str, Any]],
    observation_days: int,
    outcome_days: int,
    global_end: float,
    header_bytes: int,
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    eligible_metrics: list[dict[str, int]] = []
    positive_metrics: list[dict[str, int]] = []
    non_refund_metrics: list[dict[str, int]] = []
    for record in records:
        status = classify_anchor(
            record["first_pay"],
            record["refund_times"],
            observation_days,
            outcome_days,
            global_end,
        )
        statuses[status] += 1
        if status in {"positive", "non_refund"}:
            metric = record["metrics"][observation_days]
            eligible_metrics.append(metric)
            if status == "positive":
                positive_metrics.append(metric)
            else:
                non_refund_metrics.append(metric)

    positive = statuses["positive"]
    non_refund = statuses["non_refund"]
    eligible = positive + non_refund
    event_values = [metric["learning_event_count"] for metric in eligible_metrics]
    active_values = [metric["active_days"] for metric in eligible_metrics]
    raw_rows = sum(metric["raw_rows"] for metric in eligible_metrics)
    raw_bytes = sum(metric["raw_bytes"] for metric in eligible_metrics) + header_bytes
    return {
        "observation_days": observation_days,
        "outcome_days": outcome_days,
        "total_pay_users": len(records),
        "eligible_users": eligible,
        "refund_users": positive,
        "non_refund_users": non_refund,
        "positive_rate": positive / eligible if eligible else None,
        "refund_during_feature_excluded": statuses["refund_during_feature"],
        "observation_window_incomplete": statuses["observation_incomplete"],
        "right_censored": statuses["right_censored"],
        "feature_zero_learning_events": sum(value == 0 for value in event_values),
        "feature_fewer_than_10_learning_events": sum(value < 10 for value in event_values),
        "active_days": describe(active_values),
        "learning_event_count": describe(event_values),
        "estimated_raw_log_rows": raw_rows,
        "estimated_raw_log_bytes": raw_bytes,
        "estimated_raw_log_size": format_bytes(raw_bytes),
        "positive_raw_log_rows": sum(metric["raw_rows"] for metric in positive_metrics),
        "non_refund_raw_log_rows": sum(
            metric["raw_rows"] for metric in non_refund_metrics
        ),
        "accounting_check": (
            eligible
            + statuses["refund_during_feature"]
            + statuses["observation_incomplete"]
            + statuses["right_censored"]
            == len(records)
        ),
    }


def build_episode_result(
    records: Sequence[dict[str, Any]],
    observation_days: int,
    outcome_days: int,
    global_end: float,
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    eligible_episodes_by_user: Counter[str] = Counter()
    for record in records:
        for pay_time in record["pay_times"]:
            status = classify_anchor(
                pay_time,
                record["refund_times"],
                observation_days,
                outcome_days,
                global_end,
            )
            statuses[status] += 1
            if status in {"positive", "non_refund"}:
                eligible_episodes_by_user[record["user_id"]] += 1
    eligible = statuses["positive"] + statuses["non_refund"]
    return {
        "total_payment_episodes": sum(len(record["pay_times"]) for record in records),
        "eligible_episodes": eligible,
        "positive_episodes": statuses["positive"],
        "non_refund_episodes": statuses["non_refund"],
        "refund_during_feature_excluded": statuses["refund_during_feature"],
        "observation_window_incomplete": statuses["observation_incomplete"],
        "right_censored": statuses["right_censored"],
        "unique_users_with_eligible_episode": len(eligible_episodes_by_user),
        "users_with_multiple_eligible_episodes": sum(
            count > 1 for count in eligible_episodes_by_user.values()
        ),
        "duplicate_eligible_episodes_beyond_one_per_user": sum(
            max(0, count - 1) for count in eligible_episodes_by_user.values()
        ),
        "warning": (
            "Diagnostic only: episode windows can overlap, repeated samples from one user are "
            "correlated, and one refund can be visible to multiple anchors without a transaction ID."
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
        manifest_rows = list(csv.DictReader(handle))
    pay_manifest = {
        row["relative_path"]: row for row in manifest_rows if row["has_pay"] == "1"
    }
    refund_only_users = sum(
        row["has_pay"] == "0" and row["has_refund"] == "1" for row in manifest_rows
    )
    files = sorted(input_dir.rglob("*.csv"), key=lambda path: path.as_posix())
    print(
        f"Scanning {len(files):,} KT4 files; detailed pay users={len(pay_manifest):,}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    timestamp_units: Counter[str] = Counter()
    schemas: dict[tuple[str, ...], list[str]] = {}
    global_end: float | None = None
    non_monotonic_files = 0
    buffer = io.StringIO(newline="")
    size_writer = csv.writer(buffer)
    started = time.monotonic()

    for index, path in enumerate(files, start=1):
        relative = path.relative_to(input_dir).as_posix()
        try:
            if relative in pay_manifest:
                record, file_max, file_units, schema, header = analyze_selected_file(
                    path, pay_manifest[relative], buffer, size_writer
                )
                records.append(record)
                schemas.setdefault(schema, header)
                monotonic = record["file_timestamp_monotonic"]
            else:
                file_max, file_units, monotonic = scan_unselected_file(path)
            timestamp_units.update(file_units)
            if not monotonic:
                non_monotonic_files += 1
            if file_max is not None:
                global_end = file_max if global_end is None else max(global_end, file_max)
        except (OSError, UnicodeError, csv.Error, ValueError, StopIteration) as exc:
            failures.append(
                {
                    "source_file": relative,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        if (
            index == 1
            or index == len(files)
            or (args.progress_every > 0 and index % args.progress_every == 0)
        ):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            remaining = (len(files) - index) / rate if rate else float("inf")
            print(
                f"Analyzing pay anchors: {index:,} / {len(files):,} | "
                f"pay users={len(records):,} | {rate:.1f} files/s | "
                f"ETA {int(remaining // 60):02d}:{int(remaining % 60):02d}",
                flush=True,
            )

    if global_end is None:
        raise RuntimeError("no valid timestamp found")
    if len(records) != len(pay_manifest):
        print(
            f"WARNING: expected {len(pay_manifest):,} pay users, analyzed {len(records):,}",
            flush=True,
        )

    union_columns: list[str] = []
    seen: set[str] = set()
    for schema, header in schemas.items():
        for normalized, original in zip(schema, header, strict=True):
            if normalized in {"user_id", "datetime"}:
                continue
            if normalized not in seen:
                seen.add(normalized)
                union_columns.append(original)
    final_columns = ["user_id", *union_columns, "datetime"]
    header_bytes = serialized_size(size_writer, buffer, final_columns)

    combination_results = [
        build_combination_result(records, observation, outcome, global_end, header_bytes)
        for observation, outcome in COMBINATIONS
    ]
    episode_results = {
        f"{observation}d_{outcome}d": build_episode_result(
            records, observation, outcome, global_end
        )
        for observation, outcome in COMBINATIONS
    }

    pay_frequency = {
        "one_pay_users": sum(len(record["pay_times"]) == 1 for record in records),
        "two_pay_users": sum(len(record["pay_times"]) == 2 for record in records),
        "three_or_more_pay_users": sum(
            len(record["pay_times"]) >= 3 for record in records
        ),
        "maximum_pay_count": max(len(record["pay_times"]) for record in records),
        "total_payment_episodes": sum(len(record["pay_times"]) for record in records),
    }

    csv_fields = [
        "observation_days",
        "outcome_days",
        "eligible_users",
        "refund_users",
        "non_refund_users",
        "positive_rate",
        "refund_during_feature_excluded",
        "observation_window_incomplete",
        "right_censored",
        "feature_zero_learning_events",
        "feature_fewer_than_10_learning_events",
        "mean_active_days",
        "median_active_days",
        "mean_learning_event_count",
        "median_learning_event_count",
        "estimated_raw_log_rows",
        "estimated_raw_log_bytes",
        "estimated_raw_log_size",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in combination_results:
            writer.writerow(
                {
                    **{field: result.get(field, "") for field in csv_fields},
                    "mean_active_days": result["active_days"]["mean"],
                    "median_active_days": result["active_days"]["median"],
                    "mean_learning_event_count": result["learning_event_count"]["mean"],
                    "median_learning_event_count": result["learning_event_count"]["median"],
                }
            )

    report = {
        "creates_final_csv": False,
        "creates_feature_dataset": False,
        "creates_label_file": False,
        "performs_sampling": False,
        "analysis_unit": "one candidate per pay user anchored at first_pay",
        "classification_rules": {
            "feature_interval": "inclusive pay anchor, exclusive prediction point",
            "feature_refund": "refund inside feature interval is excluded as an early outcome",
            "positive": "refund in [prediction point, outcome end]",
            "non_refund": (
                "no refund in outcome and outcome end is not later than global data end"
            ),
            "right_censored": (
                "no observed outcome refund and outcome end is later than global data end"
            ),
        },
        "global_data_end": iso_utc(global_end),
        "global_data_end_unix_seconds": global_end,
        "pay_frequency": pay_frequency,
        "refund_only_users_excluded": refund_only_users,
        "first_pay_user_results": combination_results,
        "payment_episode_diagnostics": episode_results,
        "method_recommendation": {
            "first_pay": (
                "Preferred for the first baseline because it yields one sample per user, avoids "
                "within-user dependence, and supports ordinary user-level splitting. It can be "
                "far from a later refund and does not represent renewal-specific risk."
            ),
            "payment_episode": (
                "Potentially better for recurring-payment risk, but only after defining episode "
                "boundaries/refund linkage, preventing overlapping windows, and grouping all "
                "episodes from one user in the same train/validation fold. KT4 has no explicit "
                "transaction-refund key, so naive episode labels are not recommended yet."
            ),
        },
        "timestamp_unit_counts": dict(timestamp_units),
        "non_monotonic_source_files": non_monotonic_files,
        "source_schema_count_for_pay_users": len(schemas),
        "estimated_raw_log_columns": final_columns,
        "failures": failures,
        "artifacts": {"csv": str(output_csv), "json": str(output_json)},
    }
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n======================================")
    print("EdNet Common Pay-Anchor Analysis")
    print("======================================")
    print(f"Global data end: {iso_utc(global_end)}")
    print(
        f"Pay frequency: 1={pay_frequency['one_pay_users']:,}, "
        f"2={pay_frequency['two_pay_users']:,}, "
        f"3+={pay_frequency['three_or_more_pay_users']:,}"
    )
    for result in combination_results:
        print(
            f"Obs {result['observation_days']}d / Outcome {result['outcome_days']}d: "
            f"eligible={result['eligible_users']:,}, refund={result['refund_users']:,}, "
            f"non_refund={result['non_refund_users']:,}, "
            f"positive_rate={result['positive_rate']:.3%}, "
            f"right_censored={result['right_censored']:,}, "
            f"raw={result['estimated_raw_log_rows']:,} rows / "
            f"{result['estimated_raw_log_size']}"
        )
    print(f"Failures: {len(failures):,}")
    print(f"CSV statistics: {output_csv}")
    print(f"JSON report: {output_json}")
    print("No final/feature/label CSV was created.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
