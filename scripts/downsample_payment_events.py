#!/usr/bin/env python3
"""Create a GitHub-sized, weighted event sample without dropping paying users.

All pay/refund events are retained. Other events are selected deterministically
and evenly within three periods relative to first payment: pre-payment, the
14-day observation window, and post-observation. ``sample_weight`` records the
number of non-critical source events represented by each selected normal event.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MS_PER_DAY = 86_400_000.0
OBSERVATION_DAYS = 14
INPUT_FIELDS = [
    "user_id", "timestamp", "action_type", "item_id", "cursor_time",
    "source", "user_answer", "platform",
]
OUTPUT_FIELDS = [*INPUT_FIELDS, "sample_weight"]
CRITICAL_ACTIONS = {"enroll_coupon", "pay", "refund"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "ednet_payment_users_full.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "ednet_payment_users.csv"
    )
    parser.add_argument(
        "--summary", type=Path,
        default=DATA_DIR / "kt4_pass_expiry_repurchase_analysis.csv",
    )
    parser.add_argument(
        "--report", type=Path, default=DATA_DIR / "payment_event_sampling_report.json"
    )
    parser.add_argument("--pre-quota", type=int, default=15)
    parser.add_argument("--observation-quota", type=int, default=30)
    parser.add_argument("--post-quota", type=int, default=15)
    parser.add_argument("--max-bytes", type=int, default=95_000_000)
    parser.add_argument("--progress-every-users", type=int, default=2_000)
    return parser.parse_args()


def load_summary(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    first_pay: dict[str, float] = {}
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("has_pay") != "1":
                continue
            user_id = row["user_id"]
            first_pay[user_id] = float(row["first_pay_ts"])
            labels[user_id] = int(float(row["is_churn_overall"]))
    return first_pay, labels


def even_positions(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def segment_name(timestamp: float, first_pay: float) -> str:
    if timestamp < first_pay:
        return "pre"
    if timestamp <= first_pay + OBSERVATION_DAYS * MS_PER_DAY:
        return "observation"
    return "post"


def sample_user(
    rows: list[list[str]], first_pay: float, quotas: dict[str, int]
) -> tuple[list[list[str]], Counter[str]]:
    segments: dict[str, list[int]] = {"pre": [], "observation": [], "post": []}
    critical: set[int] = set()
    for index, row in enumerate(rows):
        timestamp = float(row[1])
        segments[segment_name(timestamp, first_pay)].append(index)
        if row[2].strip().casefold() in CRITICAL_ACTIONS:
            critical.add(index)

    selected_weights: dict[int, float] = {index: 1.0 for index in critical}
    selected_by_segment: Counter[str] = Counter()
    for name, indexes in segments.items():
        normal_indexes = [index for index in indexes if index not in critical]
        positions = even_positions(len(normal_indexes), quotas[name])
        selected_normal = [normal_indexes[position] for position in positions]
        weight = len(normal_indexes) / max(1, len(selected_normal))
        for index in selected_normal:
            selected_weights[index] = max(1.0, weight)
        selected_by_segment[name] = len(selected_normal)

    sampled: list[list[str]] = []
    for index in sorted(selected_weights):
        weight = selected_weights[index]
        weight_text = f"{weight:.6f}".rstrip("0").rstrip(".")
        sampled.append([*rows[index], weight_text])
    selected_by_segment["critical"] = len(critical)
    return sampled, selected_by_segment


def user_groups(reader: Iterable[list[str]]) -> Iterable[tuple[str, list[list[str]]]]:
    current_user: str | None = None
    rows: list[list[str]] = []
    for row in reader:
        user_id = row[0]
        if current_user is None:
            current_user = user_id
        if user_id != current_user:
            yield current_user, rows
            current_user = user_id
            rows = []
        rows.append(row)
    if current_user is not None:
        yield current_user, rows


def main() -> int:
    args = parse_args()
    quotas = {
        "pre": args.pre_quota,
        "observation": args.observation_quota,
        "post": args.post_quota,
    }
    if any(value < 0 for value in quotas.values()):
        raise ValueError("Sampling quotas cannot be negative")
    if args.max_bytes <= 0:
        raise ValueError("--max-bytes must be positive")
    if not args.input.is_file() or not args.summary.is_file():
        raise FileNotFoundError("Input raw CSV or payment summary is missing")

    first_pay, labels = load_summary(args.summary)
    started = time.monotonic()
    building = args.output.with_suffix(args.output.suffix + ".building")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    output_rows = 0
    output_users = 0
    segment_totals: Counter[str] = Counter()
    sampled_labels: Counter[int] = Counter()

    with args.input.open(
        "r", encoding="utf-8-sig", newline="", buffering=16 * 1024 * 1024
    ) as source_handle, building.open(
        "w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024
    ) as output_handle:
        reader = csv.reader(source_handle, strict=True)
        header = next(reader, None)
        if header != INPUT_FIELDS:
            raise ValueError(f"Unexpected input header: {header!r}")
        writer = csv.writer(output_handle)
        writer.writerow(OUTPUT_FIELDS)
        previous_user: str | None = None
        for user_id, rows in user_groups(reader):
            if user_id not in first_pay:
                raise ValueError(f"Unexpected non-paying user in raw input: {user_id}")
            if previous_user is not None and user_id <= previous_user:
                raise ValueError(f"Input user order violation: {user_id} after {previous_user}")
            previous_user = user_id
            source_rows += len(rows)
            sampled, counts = sample_user(rows, first_pay[user_id], quotas)
            writer.writerows(sampled)
            output_rows += len(sampled)
            output_users += 1
            segment_totals.update(counts)
            sampled_labels[labels[user_id]] += 1
            if args.progress_every_users and output_users % args.progress_every_users == 0:
                print(
                    f"Sampled {output_users:,}/{len(first_pay):,} users; "
                    f"source rows {source_rows:,}; output rows {output_rows:,}; "
                    f"elapsed {(time.monotonic() - started) / 60:.1f} min",
                    flush=True,
                )
        output_handle.flush()
        os.fsync(output_handle.fileno())

    if output_users != len(first_pay):
        building.unlink(missing_ok=True)
        raise ValueError(f"User mismatch: expected {len(first_pay):,}, sampled {output_users:,}")
    output_bytes = building.stat().st_size
    if output_bytes > args.max_bytes:
        raise ValueError(
            f"Sample is too large: {output_bytes:,} bytes > {args.max_bytes:,}. "
            "Lower one or more quotas and rerun. The oversized .building file was retained."
        )
    os.replace(building, args.output)

    report: dict[str, Any] = {
        "method": "deterministic_even_sampling_by_user_and_payment_period",
        "observation_window_days": OBSERVATION_DAYS,
        "normal_event_quotas_per_user": quotas,
        "critical_actions_always_retained": sorted(CRITICAL_ACTIONS),
        "sample_weight_semantics": (
            "1 for pay/refund/enroll_coupon; otherwise source normal-event count divided by selected "
            "normal-event count within the user's period"
        ),
        "source": {
            "path": str(args.input.resolve()), "rows": source_rows,
            "bytes": args.input.stat().st_size, "users": len(first_pay),
        },
        "sample": {
            "path": str(args.output.resolve()), "rows": output_rows,
            "bytes": args.output.stat().st_size, "users": output_users,
            "label_distribution": dict(sampled_labels),
            "selected_counts": dict(segment_totals),
        },
        "row_reduction_ratio": 1 - output_rows / source_rows,
        "byte_reduction_ratio": 1 - args.output.stat().st_size / args.input.stat().st_size,
        "max_bytes": args.max_bytes,
    }
    report_building = args.report.with_suffix(args.report.suffix + ".building")
    with report_building.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_building, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
