#!/usr/bin/env python3
"""Audit EdNet KT4 payment/refund coverage without building churn labels.

The script deliberately processes one user CSV at a time.  It never concatenates
the source files and never modifies them.  Only ``timestamp`` and ``action_type``
are parsed because the requested payment/behaviour audit does not use the other
KT4 columns.

Pay/refund matching
-------------------
Events are ordered by timestamp (and original row order for ties).  Each refund
is paired with the most recent still-unmatched pay that precedes it (LIFO).  This
avoids the misleading "first pay to last refund" shortcut and is a reasonable
choice when KT4 has no transaction/order identifier.  User-level prediction
windows use the earliest valid matched refund, so no information after that
first observable refund outcome leaks into the behavioural statistics.

"Learning behaviour" below means every action except transaction actions
``pay`` and ``refund``.  Keeping those actions out prevents a payment itself from
inflating the amount of predictive behaviour available before a refund.
"Observation days" is elapsed time (possibly fractional); "active days" is the
number of distinct UTC calendar dates containing at least one learning event.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSACTION_ACTIONS = {"pay", "refund"}
SECONDS_PER_DAY = 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream all KT4 user CSVs and audit pay/refund suitability."
    )
    parser.add_argument(
        "--kt4-dir",
        type=Path,
        default=None,
        help="KT4 directory (default: auto-detect <project>/KT4 or <project>/data/KT4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eda",
        help="Directory for compact audit artifacts",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress every N files (default: 1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional development-only file limit; omit for the required full audit",
    )
    return parser.parse_args()


def resolve_kt4_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"KT4 directory does not exist: {candidate}")
        return candidate

    candidates = (PROJECT_ROOT / "KT4", PROJECT_ROOT / "data" / "KT4")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not auto-detect KT4 directory. Tried: {attempted}")


def timestamp_to_seconds(raw: str) -> float | None:
    """Convert Unix timestamps in seconds/ms/us/ns to seconds, rejecting bad values."""
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None

    magnitude = abs(value)
    if magnitude >= 1e17:  # nanoseconds
        value /= 1e9
    elif magnitude >= 1e14:  # microseconds
        value /= 1e6
    elif magnitude >= 1e11:  # milliseconds (KT4's expected representation)
        value /= 1e3

    try:
        datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return value


def iso_utc(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


def distinct_utc_days(events: Iterable[tuple[float, int, str]]) -> int:
    return len(
        {
            datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
            for timestamp, _row_number, _action in events
        }
    )


def observation_days(events: Sequence[tuple[float, int, str]]) -> float:
    if not events:
        return 0.0
    timestamps = [event[0] for event in events]
    return (max(timestamps) - min(timestamps)) / SECONDS_PER_DAY


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def event_bin(value: int) -> str:
    if value == 0:
        return "0 events"
    if value <= 9:
        return "1-9 events"
    if value <= 49:
        return "10-49 events"
    if value <= 99:
        return "50-99 events"
    if value <= 499:
        return "100-499 events"
    return "500+ events"


def duration_bin(days: float) -> str:
    hours = days * 24.0
    if hours < 1:
        return "< 1 hour"
    if hours < 24:
        return "1-24 hours"
    if days < 3:
        return "1-3 days"
    if days < 7:
        return "3-7 days"
    if days < 14:
        return "7-14 days"
    if days < 30:
        return "14-30 days"
    return "30+ days"


def read_user_file(path: Path, user_id: str) -> tuple[dict[str, Any], Counter[str], set[str]]:
    """Read and summarize exactly one user CSV; raise on file/schema errors."""
    events: list[tuple[float, int, str]] = []
    action_counts: Counter[str] = Counter()
    actions_seen: set[str] = set()
    total_rows = 0
    invalid_timestamp_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("empty CSV (no header)") from exc

        normalized_header = [name.strip().lower() for name in header]
        missing = {"timestamp", "action_type"} - set(normalized_header)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        timestamp_index = normalized_header.index("timestamp")
        action_index = normalized_header.index("action_type")
        required_index = max(timestamp_index, action_index)

        for row_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) <= required_index:
                raise ValueError(
                    f"row {row_number} has {len(row)} columns; needs index {required_index}"
                )
            total_rows += 1
            action = row[action_index].strip() or "<MISSING>"
            action_counts[action] += 1
            actions_seen.add(action)
            timestamp = timestamp_to_seconds(row[timestamp_index])
            if timestamp is None:
                invalid_timestamp_count += 1
                continue
            events.append((timestamp, row_number, action))

    events.sort(key=lambda item: (item[0], item[1]))
    pay_events = [event for event in events if event[2].casefold() == "pay"]
    refund_events = [event for event in events if event[2].casefold() == "refund"]
    learning_events = [
        event for event in events if event[2].casefold() not in TRANSACTION_ACTIONS
    ]

    unmatched_pays: list[tuple[float, int, str]] = []
    matched_pairs: list[
        tuple[tuple[float, int, str], tuple[float, int, str]]
    ] = []
    for event in events:
        normalized_action = event[2].casefold()
        if normalized_action == "pay":
            unmatched_pays.append(event)
        elif normalized_action == "refund" and unmatched_pays:
            # Most recent unmatched preceding pay, because KT4 has no order ID.
            matched_pairs.append((unmatched_pays.pop(), event))

    matched_pairs.sort(key=lambda pair: (pair[1][0], pair[1][1]))
    first_valid_pair = matched_pairs[0] if matched_pairs else None
    has_pay = bool(pay_events)
    has_refund = bool(refund_events)

    if first_valid_pair:
        payment_status = "pay_then_refund"
    elif has_pay and has_refund:
        payment_status = "pay_and_refund_no_valid_order"
    elif has_pay:
        payment_status = "pay_only"
    elif has_refund:
        payment_status = "refund_only"
    else:
        payment_status = "neither"

    first_pay = pay_events[0][0] if pay_events else None
    first_refund = refund_events[0][0] if refund_events else None
    first_learning = learning_events[0][0] if learning_events else None
    last_learning = learning_events[-1][0] if learning_events else None
    last_event = events[-1][0] if events else None

    before_refund: list[tuple[float, int, str]] = []
    between_pay_refund: list[tuple[float, int, str]] = []
    matched_pay_timestamp: float | None = None
    refund_after_pay_timestamp: float | None = None
    pay_to_refund_days: float | None = None
    if first_valid_pair:
        matched_pay_timestamp = first_valid_pair[0][0]
        refund_after_pay_timestamp = first_valid_pair[1][0]
        pay_to_refund_days = (
            refund_after_pay_timestamp - matched_pay_timestamp
        ) / SECONDS_PER_DAY
        before_refund = [
            event for event in learning_events if event[0] < refund_after_pay_timestamp
        ]
        between_pay_refund = [
            event
            for event in learning_events
            if matched_pay_timestamp < event[0] < refund_after_pay_timestamp
        ]

    after_first_pay: list[tuple[float, int, str]] = []
    if first_pay is not None:
        after_first_pay = [event for event in learning_events if event[0] > first_pay]

    summary = {
        "user_id": user_id,
        "source_file": path.name,
        "total_event_count": total_rows,
        "valid_timestamp_count": len(events),
        "invalid_timestamp_count": invalid_timestamp_count,
        "has_pay": int(has_pay),
        "has_refund": int(has_refund),
        "pay_count": len(pay_events),
        "refund_count": len(refund_events),
        "payment_status": payment_status,
        "has_pay_then_refund": int(first_valid_pair is not None),
        "matched_pay_refund_pair_count": len(matched_pairs),
        "first_pay_timestamp": iso_utc(first_pay),
        "first_refund_timestamp": iso_utc(first_refund),
        "matched_pay_timestamp": iso_utc(matched_pay_timestamp),
        "first_refund_after_pay": iso_utc(refund_after_pay_timestamp),
        "pay_to_refund_hours": (
            pay_to_refund_days * 24.0 if pay_to_refund_days is not None else ""
        ),
        "pay_to_refund_days": pay_to_refund_days if pay_to_refund_days is not None else "",
        "first_activity_timestamp": iso_utc(
            before_refund[0][0] if before_refund else None
        ),
        "last_activity_before_refund": iso_utc(
            before_refund[-1][0] if before_refund else None
        ),
        "events_before_refund": len(before_refund),
        "active_days_before_refund": distinct_utc_days(before_refund),
        "observation_days_before_refund": observation_days(before_refund),
        "events_between_pay_refund": len(between_pay_refund),
        "active_days_between_pay_refund": distinct_utc_days(between_pay_refund),
        "observation_days_between_pay_refund": observation_days(between_pay_refund),
        "first_learning_activity_timestamp": iso_utc(first_learning),
        "last_activity_timestamp": iso_utc(last_learning),
        "last_event_timestamp": iso_utc(last_event),
        "pay_only_events_after_pay": len(after_first_pay) if payment_status == "pay_only" else "",
        "pay_only_active_days_after_pay": (
            distinct_utc_days(after_first_pay) if payment_status == "pay_only" else ""
        ),
        "pay_only_observation_days_after_pay": (
            observation_days(after_first_pay) if payment_status == "pay_only" else ""
        ),
        # Private numeric values are removed before writing the CSV.
        "_global_min_timestamp": events[0][0] if events else None,
        "_global_max_timestamp": events[-1][0] if events else None,
        "_first_pay_seconds": first_pay,
        "_pay_to_refund_days": pay_to_refund_days,
        "_before_refund_events": before_refund,
        "_between_pay_refund": between_pay_refund,
    }
    return summary, action_counts, actions_seen


BASE_SUMMARY_FIELDS = [
    "user_id",
    "source_file",
    "total_event_count",
    "valid_timestamp_count",
    "invalid_timestamp_count",
    "has_pay",
    "has_refund",
    "pay_count",
    "refund_count",
    "payment_status",
    "has_pay_then_refund",
    "matched_pay_refund_pair_count",
    "first_pay_timestamp",
    "first_refund_timestamp",
    "matched_pay_timestamp",
    "first_refund_after_pay",
    "pay_to_refund_hours",
    "pay_to_refund_days",
    "first_activity_timestamp",
    "last_activity_before_refund",
    "events_before_refund",
    "active_days_before_refund",
    "observation_days_before_refund",
    "events_between_pay_refund",
    "active_days_between_pay_refund",
    "observation_days_between_pay_refund",
    "first_learning_activity_timestamp",
    "last_activity_timestamp",
    "last_event_timestamp",
    "pay_only_events_after_pay",
    "pay_only_active_days_after_pay",
    "pay_only_observation_days_after_pay",
]

FINAL_SUMMARY_FIELDS = BASE_SUMMARY_FIELDS + [
    "first_pay_days_before_data_end",
    "first_pay_within_1_day_of_data_end",
    "first_pay_within_3_days_of_data_end",
    "first_pay_within_7_days_of_data_end",
    "first_pay_within_14_days_of_data_end",
    "first_pay_within_30_days_of_data_end",
]


def write_action_summary(
    path: Path, event_counts: Counter[str], user_counts: Counter[str]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["action_type", "event_count", "unique_user_count"]
        )
        writer.writeheader()
        for action, count in sorted(
            event_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            writer.writerow(
                {
                    "action_type": action,
                    "event_count": count,
                    "unique_user_count": user_counts[action],
                }
            )


def enrich_summary_with_cutoff(
    raw_path: Path,
    final_path: Path,
    global_end: float | None,
    pay_only_cutoff_counts: Counter[str],
) -> None:
    with raw_path.open("r", encoding="utf-8-sig", newline="") as source, final_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=FINAL_SUMMARY_FIELDS)
        writer.writeheader()
        for row in reader:
            first_pay_raw = row.pop("_first_pay_seconds", "")
            days_before_end: float | str = ""
            if global_end is not None and first_pay_raw:
                days_before_end = (global_end - float(first_pay_raw)) / SECONDS_PER_DAY
            row["first_pay_days_before_data_end"] = days_before_end
            is_pay_only = row["payment_status"] == "pay_only"
            for days in (1, 3, 7, 14, 30):
                flag = int(
                    is_pay_only
                    and days_before_end != ""
                    and 0 <= float(days_before_end) <= days
                )
                row[f"first_pay_within_{days}_day{'s' if days != 1 else ''}_of_data_end"] = flag
                if flag:
                    pay_only_cutoff_counts[f"within_{days}_days"] += 1
            writer.writerow({field: row.get(field, "") for field in FINAL_SUMMARY_FIELDS})


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_int = int(seconds)
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_report(report: dict[str, Any]) -> None:
    overall = report["overall"]
    payment = report["payment_users"]
    duration = report["pay_to_refund_duration"]
    interval = report["between_pay_and_refund_learning_behavior"]
    print("\n=== KT4 PAYMENT/REFUND AUDIT ===")
    print("\n[전체 데이터]")
    print(f"전체 사용자(정상 처리): {overall['successful_files']:,}")
    print(f"전체 이벤트: {overall['total_events']:,}")
    print(
        f"전체 관찰 기간: {overall['first_timestamp']} ~ {overall['last_timestamp']} "
        f"({overall['observation_days']:.3f} days)"
        if overall["observation_days"] is not None
        else "전체 관찰 기간: 유효 timestamp 없음"
    )
    print(f"정상 처리 파일: {overall['successful_files']:,}")
    print(f"실패 파일: {overall['failed_files']:,}")
    print("\n[결제]")
    print(f"Pay 사용자: {payment['pay_users']:,}")
    print(f"Refund 사용자: {payment['refund_users']:,}")
    print(f"Pay + Refund 사용자: {payment['pay_and_refund_users']:,}")
    print(f"Pay -> Refund 사용자: {payment['pay_then_refund_users']:,}")
    print(f"Pay Only 사용자: {payment['pay_only_users']:,}")
    print(f"Refund Only 사용자: {payment['refund_only_users']:,}")
    print("\n[Pay -> Refund]")
    for label, key in (
        ("평균", "mean"),
        ("중앙값", "median"),
        ("Q1", "q1"),
        ("Q3", "q3"),
        ("최소", "min"),
        ("최대", "max"),
    ):
        days = duration["days"][key]
        hours = duration["hours"][key]
        print(
            f"{label}: {hours:.3f} hours / {days:.3f} days"
            if days is not None
            else f"{label}: N/A"
        )
    print("\n[환불 전 행동]")
    for key, value in report["learning_events_before_refund_bins"].items():
        print(f"{key}: {value:,}")
    print("\n[환불 전 관찰기간]")
    for key, value in report["observation_days_before_refund_thresholds"].items():
        print(f"{key}: {value:,}")
    print("\n[Pay -> Refund 사이 행동]")
    print(f"Pay -> Refund 사용자 수: {interval['users']:,}")
    print(f"행동 1개 이상 사용자 수: {interval['users_with_at_least_one_event']:,}")
    print(f"평균 event 수: {interval['event_count']['mean']}")
    print(f"중앙 event 수: {interval['event_count']['median']}")
    print(f"평균 active_days: {interval['active_days']['mean']}")
    print(f"평균 observation_days: {interval['observation_days']['mean']}")
    print("\n[Action Type]")
    print(f"{'action_type':<28} {'event_count':>15} {'unique_user_count':>20}")
    for row in report["action_types"]:
        print(
            f"{row['action_type']:<28} {row['event_count']:>15,} "
            f"{row['unique_user_count']:>20,}"
        )


def main() -> int:
    args = parse_args()
    kt4_dir = resolve_kt4_dir(args.kt4_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"KT4 directory: {kt4_dir}", flush=True)
    print("Discovering CSV files...", flush=True)
    files = sorted(kt4_dir.rglob("*.csv"), key=lambda path: path.as_posix())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        files = files[: args.limit]
        print(f"WARNING: development limit enabled ({args.limit:,} files).", flush=True)
    total_files = len(files)
    if total_files == 0:
        raise FileNotFoundError(f"No CSV files found below {kt4_dir}")
    print(f"Discovered {total_files:,} CSV files.", flush=True)

    raw_summary_path = output_dir / "kt4_user_payment_summary.partial.csv"
    final_summary_tmp = output_dir / "kt4_user_payment_summary.tmp.csv"
    final_summary_path = output_dir / "kt4_user_payment_summary.csv"
    action_summary_path = output_dir / "kt4_action_type_summary.csv"
    errors_path = output_dir / "kt4_payment_analysis_errors.csv"
    report_path = output_dir / "kt4_payment_analysis_report.json"

    action_event_counts: Counter[str] = Counter()
    action_user_counts: Counter[str] = Counter()
    pre_refund_action_event_counts: Counter[str] = Counter()
    pre_refund_action_user_counts: Counter[str] = Counter()
    payment_status_counts: Counter[str] = Counter()
    pre_refund_event_bins: Counter[str] = Counter()
    refund_duration_bins: Counter[str] = Counter()
    before_refund_observation_thresholds: Counter[str] = Counter()
    before_refund_active_thresholds: Counter[str] = Counter()
    failures: list[dict[str, str]] = []

    pay_to_refund_days_values: list[float] = []
    before_refund_events_values: list[float] = []
    before_refund_active_values: list[float] = []
    before_refund_observation_values: list[float] = []
    between_events_values: list[float] = []
    between_active_values: list[float] = []
    between_observation_values: list[float] = []
    pay_only_after_events_values: list[float] = []
    pay_only_after_active_values: list[float] = []
    pay_only_after_observation_values: list[float] = []

    global_min: float | None = None
    global_max: float | None = None
    total_events = 0
    invalid_timestamps = 0
    successful_files = 0
    pay_users = 0
    refund_users = 0
    pay_and_refund_users = 0
    started = time.monotonic()

    raw_fields = BASE_SUMMARY_FIELDS + ["_first_pay_seconds"]
    with raw_summary_path.open("w", encoding="utf-8-sig", newline="") as summary_handle:
        summary_writer = csv.DictWriter(summary_handle, fieldnames=raw_fields)
        summary_writer.writeheader()

        for index, path in enumerate(files, start=1):
            relative = path.relative_to(kt4_dir)
            user_id = relative.with_suffix("").as_posix()
            try:
                summary, file_action_counts, file_actions = read_user_file(path, user_id)
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                failures.append(
                    {
                        "source_file": relative.as_posix(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            else:
                successful_files += 1
                total_events += int(summary["total_event_count"])
                invalid_timestamps += int(summary["invalid_timestamp_count"])
                action_event_counts.update(file_action_counts)
                action_user_counts.update(file_actions)
                status = str(summary["payment_status"])
                payment_status_counts[status] += 1
                pay_users += int(summary["has_pay"])
                refund_users += int(summary["has_refund"])
                pay_and_refund_users += int(
                    bool(summary["has_pay"]) and bool(summary["has_refund"])
                )

                file_min = summary.pop("_global_min_timestamp")
                file_max = summary.pop("_global_max_timestamp")
                if file_min is not None:
                    global_min = file_min if global_min is None else min(global_min, file_min)
                if file_max is not None:
                    global_max = file_max if global_max is None else max(global_max, file_max)

                pay_to_refund_days = summary.pop("_pay_to_refund_days")
                before_refund = summary.pop("_before_refund_events")
                between = summary.pop("_between_pay_refund")
                if pay_to_refund_days is not None:
                    pay_to_refund_days_values.append(float(pay_to_refund_days))
                    refund_duration_bins[duration_bin(float(pay_to_refund_days))] += 1
                    pre_refund_event_bins[event_bin(len(before_refund))] += 1
                    before_observation = float(summary["observation_days_before_refund"])
                    before_active = int(summary["active_days_before_refund"])
                    before_refund_events_values.append(float(len(before_refund)))
                    before_refund_active_values.append(float(before_active))
                    before_refund_observation_values.append(before_observation)
                    between_events_values.append(float(len(between)))
                    between_active_values.append(
                        float(summary["active_days_between_pay_refund"])
                    )
                    between_observation_values.append(
                        float(summary["observation_days_between_pay_refund"])
                    )
                    for days in (1, 3, 7, 14, 30):
                        if before_observation >= days:
                            before_refund_observation_thresholds[f">= {days} days"] += 1
                    for days in (1, 3, 7, 14):
                        if before_active >= days:
                            before_refund_active_thresholds[f">= {days} active days"] += 1
                    pre_refund_actions_for_user = {event[2] for event in before_refund}
                    pre_refund_action_user_counts.update(pre_refund_actions_for_user)
                    pre_refund_action_event_counts.update(event[2] for event in before_refund)

                if status == "pay_only":
                    pay_only_after_events_values.append(
                        float(summary["pay_only_events_after_pay"])
                    )
                    pay_only_after_active_values.append(
                        float(summary["pay_only_active_days_after_pay"])
                    )
                    pay_only_after_observation_values.append(
                        float(summary["pay_only_observation_days_after_pay"])
                    )

                summary_writer.writerow(
                    {field: summary.get(field, "") for field in raw_fields}
                )

            should_report = (
                index == 1
                or index == total_files
                or (args.progress_every > 0 and index % args.progress_every == 0)
            )
            if should_report:
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                remaining = (total_files - index) / rate if rate else float("inf")
                print(
                    f"Processed {index:,} / {total_files:,} | "
                    f"ok={successful_files:,} failed={len(failures):,} | "
                    f"{rate:.1f} files/s | ETA {format_eta(remaining)}",
                    flush=True,
                )

    pay_only_cutoff_counts: Counter[str] = Counter()
    enrich_summary_with_cutoff(
        raw_summary_path, final_summary_tmp, global_max, pay_only_cutoff_counts
    )
    os.replace(final_summary_tmp, final_summary_path)
    raw_summary_path.unlink(missing_ok=True)
    write_action_summary(action_summary_path, action_event_counts, action_user_counts)

    with errors_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source_file", "error_type", "error_message"]
        )
        writer.writeheader()
        writer.writerows(failures)

    duration_days = describe(pay_to_refund_days_values)
    duration_hours = describe([value * 24.0 for value in pay_to_refund_days_values])
    action_rows = [
        {
            "action_type": action,
            "event_count": count,
            "unique_user_count": action_user_counts[action],
        }
        for action, count in sorted(
            action_event_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    pre_refund_action_rows = [
        {
            "action_type": action,
            "event_count": count,
            "unique_pay_then_refund_user_count": pre_refund_action_user_counts[action],
        }
        for action, count in sorted(
            pre_refund_action_event_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    observation_days_total = (
        (global_max - global_min) / SECONDS_PER_DAY
        if global_min is not None and global_max is not None
        else None
    )
    report: dict[str, Any] = {
        "scope": {
            "kt4_directory": str(kt4_dir),
            "source_files_are_modified": False,
            "learning_behavior_definition": "all action types except pay and refund",
            "pay_refund_matching": (
                "chronological LIFO: each refund is matched to the most recent "
                "unmatched preceding pay; user windows use the earliest valid matched refund"
            ),
            "timestamp_handling": "Unix seconds/ms/us/ns auto-detected; output in UTC",
        },
        "overall": {
            "discovered_csv_files": total_files,
            "successful_files": successful_files,
            "failed_files": len(failures),
            "total_events": total_events,
            "events_with_invalid_timestamp": invalid_timestamps,
            "first_timestamp": iso_utc(global_min),
            "last_timestamp": iso_utc(global_max),
            "observation_days": observation_days_total,
        },
        "payment_users": {
            "pay_users": pay_users,
            "refund_users": refund_users,
            "pay_and_refund_users": pay_and_refund_users,
            "pay_then_refund_users": payment_status_counts["pay_then_refund"],
            "pay_and_refund_no_valid_order_users": payment_status_counts[
                "pay_and_refund_no_valid_order"
            ],
            "pay_only_users": payment_status_counts["pay_only"],
            "refund_only_users": payment_status_counts["refund_only"],
            "neither_users": payment_status_counts["neither"],
        },
        "pay_to_refund_duration": {
            "days": duration_days,
            "hours": duration_hours,
            "user_bins": {
                label: refund_duration_bins[label]
                for label in (
                    "< 1 hour",
                    "1-24 hours",
                    "1-3 days",
                    "3-7 days",
                    "7-14 days",
                    "14-30 days",
                    "30+ days",
                )
            },
        },
        "learning_events_before_refund_bins": {
            label: pre_refund_event_bins[label]
            for label in (
                "0 events",
                "1-9 events",
                "10-49 events",
                "50-99 events",
                "100-499 events",
                "500+ events",
            )
        },
        "learning_behavior_before_refund": {
            "event_count": describe(before_refund_events_values),
            "active_days": describe(before_refund_active_values),
            "observation_days": describe(before_refund_observation_values),
        },
        "observation_days_before_refund_thresholds": {
            f">= {days} days": before_refund_observation_thresholds[f">= {days} days"]
            for days in (1, 3, 7, 14, 30)
        },
        "active_days_before_refund_thresholds": {
            f">= {days} active days": before_refund_active_thresholds[
                f">= {days} active days"
            ]
            for days in (1, 3, 7, 14)
        },
        "between_pay_and_refund_learning_behavior": {
            "users": len(between_events_values),
            "users_with_at_least_one_event": sum(
                value > 0 for value in between_events_values
            ),
            "event_count": describe(between_events_values),
            "active_days": describe(between_active_values),
            "observation_days": describe(between_observation_values),
        },
        "pay_only_after_pay_learning_behavior": {
            "users": len(pay_only_after_events_values),
            "users_with_at_least_one_event": sum(
                value > 0 for value in pay_only_after_events_values
            ),
            "event_count": describe(pay_only_after_events_values),
            "active_days": describe(pay_only_after_active_values),
            "observation_days": describe(pay_only_after_observation_values),
        },
        "pay_only_first_pay_near_data_end": {
            f"within_{days}_days": pay_only_cutoff_counts[f"within_{days}_days"]
            for days in (1, 3, 7, 14, 30)
        },
        "action_types": action_rows,
        "actions_before_first_valid_refund": pre_refund_action_rows,
        "failures": failures,
        "artifacts": {
            "user_summary": str(final_summary_path),
            "action_type_summary": str(action_summary_path),
            "errors": str(errors_path),
        },
    }
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print_report(report)
    print("\nArtifacts:")
    print(f"- {final_summary_path}")
    print(f"- {action_summary_path}")
    print(f"- {errors_path}")
    print(f"- {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; source KT4 files were not modified.", file=sys.stderr)
        raise SystemExit(130)
