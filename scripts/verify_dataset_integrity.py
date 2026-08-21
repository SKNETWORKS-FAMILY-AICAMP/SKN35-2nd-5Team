#!/usr/bin/env python3
"""Independently verify consolidated KT4 payment-user datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPECTED_USERS = 23_789
EXPECTED_EVENTS = 72_875_338
EXPECTED_CHURN = {0: 2_174, 1: 21_615}
FORBIDDEN_FUTURE_FEATURES = {
    "last_pay_item", "pay_count", "refund_count", "has_refund", "has_repurchase"
}
RAW_FIELDS = [
    "user_id", "timestamp", "action_type", "item_id", "cursor_time",
    "source", "user_answer", "platform",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DATA_DIR / "ednet_payment_users_full.csv")
    parser.add_argument("--features", type=Path, default=DATA_DIR / "churn_modeling_features.csv")
    parser.add_argument(
        "--summary", type=Path,
        default=DATA_DIR / "kt4_pass_expiry_repurchase_analysis.csv",
    )
    parser.add_argument(
        "--transactions", type=Path,
        default=DATA_DIR / "kt4_payment_transactions.csv",
    )
    parser.add_argument(
        "--report", type=Path, default=DATA_DIR / "dataset_integrity_report.json"
    )
    parser.add_argument("--expected-users", type=int, default=EXPECTED_USERS)
    parser.add_argument("--expected-events", type=int, default=EXPECTED_EVENTS)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Development-only: verify the first N paying users in lexical order.",
    )
    parser.add_argument(
        "--allow-noncanonical-counts", action="store_true",
        help="Use supplied user/event counts without enforcing the canonical churn distribution.",
    )
    return parser.parse_args()


def load_expected(
    summary_path: Path, limit: int | None
) -> tuple[dict[str, int], dict[str, int]]:
    rows: list[dict[str, str]] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("has_pay") == "1":
                rows.append(row)
    rows.sort(key=lambda row: row["user_id"])
    if limit is not None:
        rows = rows[:limit]
    event_counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    for row in rows:
            user_id = row["user_id"]
            event_counts[user_id] = int(float(row["total_events"]))
            labels[user_id] = int(float(row["is_churn_overall"]))
    return event_counts, labels


def load_features(path: Path) -> tuple[list[str], dict[str, int], list[str]]:
    user_ids: list[str] = []
    labels: dict[str, int] = {}
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {"user_id", "is_churn"}
        if not required.issubset(columns):
            raise ValueError(f"Feature columns missing: {sorted(required - set(columns))}")
        for row_number, row in enumerate(reader, start=2):
            user_id = row["user_id"]
            if user_id in labels:
                raise ValueError(f"Duplicate feature user {user_id!r} at row {row_number}")
            user_ids.append(user_id)
            labels[user_id] = int(float(row["is_churn"]))
    return user_ids, labels, columns


def load_transaction_pay_users(path: Path) -> set[str]:
    pay_users: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("action_type", "").strip().casefold() == "pay":
                pay_users.add(row["user_id"])
    return pay_users


def validate_raw(
    path: Path, expected_counts: dict[str, int], progress_every: int
) -> dict[str, Any]:
    started = time.monotonic()
    actual_counts: Counter[str] = Counter()
    pay_users: set[str] = set()
    total_rows = 0
    previous_user: str | None = None
    previous_timestamp: float | None = None
    closed_users: set[str] = set()
    order_errors: list[str] = []
    invalid_rows: list[str] = []
    with path.open(
        "r", encoding="utf-8-sig", newline="", buffering=16 * 1024 * 1024
    ) as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header != RAW_FIELDS:
            raise ValueError(f"Unexpected raw header: {header!r}")
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            if len(row) != len(RAW_FIELDS):
                if len(invalid_rows) < 20:
                    invalid_rows.append(
                        f"row {row_number}: expected {len(RAW_FIELDS)} columns, got {len(row)}"
                    )
                continue
            user_id = row[0]
            try:
                timestamp = float(row[1])
                if not math.isfinite(timestamp):
                    raise ValueError
            except ValueError:
                if len(invalid_rows) < 20:
                    invalid_rows.append(f"row {row_number}: invalid timestamp {row[1]!r}")
                continue
            if user_id != previous_user:
                if previous_user is not None:
                    closed_users.add(previous_user)
                    if user_id <= previous_user and len(order_errors) < 20:
                        order_errors.append(
                            f"row {row_number}: user order {user_id!r} after {previous_user!r}"
                        )
                if user_id in closed_users and len(order_errors) < 20:
                    order_errors.append(f"row {row_number}: non-contiguous user {user_id!r}")
                previous_user = user_id
                previous_timestamp = None
            if previous_timestamp is not None and timestamp < previous_timestamp:
                if len(order_errors) < 20:
                    order_errors.append(
                        f"row {row_number}: timestamp {timestamp} after {previous_timestamp} for {user_id}"
                    )
            previous_timestamp = timestamp
            actual_counts[user_id] += 1
            if row[2].strip().casefold() == "pay":
                pay_users.add(user_id)
            if progress_every > 0 and total_rows % progress_every == 0:
                print(
                    f"Verified {total_rows:,} raw rows; users {len(actual_counts):,}; "
                    f"elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True,
                )

    mismatches = {
        user_id: {"expected": expected, "actual": actual_counts.get(user_id, 0)}
        for user_id, expected in expected_counts.items()
        if actual_counts.get(user_id, 0) != expected
    }
    unexpected_users = sorted(set(actual_counts) - set(expected_counts))
    return {
        "row_count": total_rows,
        "unique_user_count": len(actual_counts),
        "pay_action_user_count": len(pay_users),
        "invalid_row_count": len(invalid_rows),
        "order_error_count": len(order_errors),
        "row_count_mismatch_count": len(mismatches),
        "unexpected_user_count": len(unexpected_users),
        "sample_invalid_rows": invalid_rows,
        "sample_order_errors": order_errors,
        "sample_row_count_mismatches": dict(list(mismatches.items())[:20]),
        "sample_unexpected_users": unexpected_users[:20],
        "pay_users": pay_users,
        "actual_users": set(actual_counts),
    }


def main() -> int:
    args = parse_args()
    paths = [args.raw, args.features, args.summary, args.transactions]
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    expected_counts, expected_labels = load_expected(args.summary, args.limit)
    feature_ids, feature_labels, feature_columns = load_features(args.features)
    transaction_pay_users = load_transaction_pay_users(args.transactions)
    raw = validate_raw(args.raw, expected_counts, args.progress_every)
    raw_pay_users = raw.pop("pay_users")
    raw_users = raw.pop("actual_users")
    expected_users = set(expected_counts)
    if args.limit is not None:
        transaction_pay_users &= expected_users
    feature_users = set(feature_ids)
    expected_churn = Counter(expected_labels.values())
    feature_churn = Counter(feature_labels.values())
    label_mismatches = {
        user_id: {"expected": expected_labels[user_id], "actual": feature_labels.get(user_id)}
        for user_id in expected_users
        if feature_labels.get(user_id) != expected_labels[user_id]
    }
    canonical_churn_ok = (
        args.allow_noncanonical_counts or dict(expected_churn) == EXPECTED_CHURN
    )
    feature_churn_ok = (
        args.allow_noncanonical_counts or dict(feature_churn) == EXPECTED_CHURN
    )
    checks = {
        "summary_user_count": len(expected_users) == args.expected_users,
        "summary_event_count": sum(expected_counts.values()) == args.expected_events,
        "summary_churn_distribution": canonical_churn_ok,
        "feature_user_count": len(feature_ids) == args.expected_users,
        "feature_unique_users": len(feature_users) == len(feature_ids),
        "feature_user_order": feature_ids == sorted(feature_ids),
        "feature_user_set_matches_summary": feature_users == expected_users,
        "feature_labels_match_summary": not label_mismatches,
        "feature_churn_distribution": feature_churn_ok,
        "feature_has_no_future_summary_leakage": not (
            set(feature_columns) & FORBIDDEN_FUTURE_FEATURES
        ),
        "raw_event_count": raw["row_count"] == args.expected_events,
        "raw_user_count": raw["unique_user_count"] == args.expected_users,
        "raw_user_set_matches_summary": raw_users == expected_users,
        "raw_all_users_have_pay_action": raw_pay_users == expected_users,
        "transaction_pay_users_match_summary": transaction_pay_users == expected_users,
        "raw_rows_well_formed": raw["invalid_row_count"] == 0,
        "raw_sorted_by_user_and_timestamp": raw["order_error_count"] == 0,
        "raw_per_user_counts_match_summary": raw["row_count_mismatch_count"] == 0,
        "raw_has_no_unexpected_users": raw["unexpected_user_count"] == 0,
    }
    passed = all(checks.values())
    report = {
        "passed": passed,
        "checks": checks,
        "expected": {
            "users": args.expected_users, "events": args.expected_events,
            "churn_distribution": EXPECTED_CHURN,
        },
        "actual": {
            "summary_users": len(expected_users),
            "summary_events": sum(expected_counts.values()),
            "summary_churn_distribution": dict(expected_churn),
            "feature_users": len(feature_ids), "feature_columns": len(feature_columns),
            "feature_churn_distribution": dict(feature_churn), "raw": raw,
            "transaction_pay_users": len(transaction_pay_users),
            "raw_file_bytes": args.raw.stat().st_size,
            "feature_file_bytes": args.features.stat().st_size,
        },
        "label_mismatch_count": len(label_mismatches),
        "sample_label_mismatches": dict(list(label_mismatches.items())[:20]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_building = args.report.with_suffix(args.report.suffix + ".building")
    with report_building.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_building, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
