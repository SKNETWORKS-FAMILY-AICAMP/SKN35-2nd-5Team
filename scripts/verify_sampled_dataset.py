#!/usr/bin/env python3
"""Verify the GitHub-sized weighted KT4 payment-event sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPECTED_USERS = 23_789
EXPECTED_LABELS = {0: 2_174, 1: 21_615}
FIELDS = [
    "user_id", "timestamp", "action_type", "item_id", "cursor_time", "source",
    "user_answer", "platform", "sample_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=DATA_DIR / "ednet_payment_users_sampled.csv"
    )
    parser.add_argument(
        "--features", type=Path, default=DATA_DIR / "churn_modeling_features.csv"
    )
    parser.add_argument(
        "--transactions", type=Path, default=DATA_DIR / "kt4_payment_transactions.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=DATA_DIR / "sampled_dataset_integrity_report.json"
    )
    parser.add_argument("--max-bytes", type=int, default=95_000_000)
    return parser.parse_args()


def transaction_key(row: dict[str, str]) -> tuple[str, float, str, str]:
    return (
        row["user_id"], float(row["timestamp"]),
        row["action_type"].strip().casefold(), row["item_id"],
    )


def load_features(path: Path) -> tuple[list[str], Counter[int]]:
    users: list[str] = []
    labels: Counter[int] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            users.append(row["user_id"])
            labels[int(float(row["is_churn"]))] += 1
    return users, labels


def main() -> int:
    args = parse_args()
    feature_users, labels = load_features(args.features)
    expected_users = set(feature_users)
    expected_transactions: Counter[tuple[str, float, str, str]] = Counter()
    with args.transactions.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["user_id"] in expected_users:
                expected_transactions[transaction_key(row)] += 1

    actual_transactions: Counter[tuple[str, float, str, str]] = Counter()
    actual_users: set[str] = set()
    user_rows: Counter[str] = Counter()
    row_count = 0
    invalid_rows = 0
    order_errors = 0
    invalid_weights = 0
    previous_user: str | None = None
    previous_timestamp: float | None = None
    with args.sample.open(
        "r", encoding="utf-8-sig", newline="", buffering=16 * 1024 * 1024
    ) as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header != FIELDS:
            raise ValueError(f"Unexpected sample header: {header!r}")
        for row in reader:
            row_count += 1
            if len(row) != len(FIELDS):
                invalid_rows += 1
                continue
            user_id = row[0]
            try:
                timestamp = float(row[1])
                weight = float(row[8])
                if not math.isfinite(timestamp) or not math.isfinite(weight):
                    raise ValueError
            except ValueError:
                invalid_rows += 1
                continue
            if weight < 1:
                invalid_weights += 1
            if user_id != previous_user:
                if previous_user is not None and user_id <= previous_user:
                    order_errors += 1
                previous_user = user_id
                previous_timestamp = None
            if previous_timestamp is not None and timestamp < previous_timestamp:
                order_errors += 1
            previous_timestamp = timestamp
            actual_users.add(user_id)
            user_rows[user_id] += 1
            action = row[2].strip().casefold()
            if action in {"enroll_coupon", "pay", "refund"}:
                actual_transactions[(user_id, timestamp, action, row[3])] += 1

    missing_transactions = expected_transactions - actual_transactions
    unexpected_transactions = actual_transactions - expected_transactions
    checks = {
        "file_is_below_max_bytes": args.sample.stat().st_size <= args.max_bytes,
        "feature_user_count": len(feature_users) == EXPECTED_USERS,
        "feature_users_unique": len(expected_users) == len(feature_users),
        "feature_label_distribution": dict(labels) == EXPECTED_LABELS,
        "all_feature_users_in_sample": actual_users == expected_users,
        "every_user_has_sampled_rows": all(user_rows[user] > 0 for user in expected_users),
        "rows_well_formed": invalid_rows == 0,
        "weights_valid": invalid_weights == 0,
        "sorted_by_user_and_timestamp": order_errors == 0,
        "all_payment_transactions_retained": not missing_transactions,
        "no_unexpected_payment_transactions": not unexpected_transactions,
    }
    report: dict[str, Any] = {
        "passed": all(checks.values()), "checks": checks,
        "sample_file_bytes": args.sample.stat().st_size, "sample_rows": row_count,
        "sample_users": len(actual_users), "feature_label_distribution": dict(labels),
        "expected_payment_transactions": sum(expected_transactions.values()),
        "actual_payment_transactions": sum(actual_transactions.values()),
        "missing_payment_transaction_count": sum(missing_transactions.values()),
        "unexpected_payment_transaction_count": sum(unexpected_transactions.values()),
        "invalid_row_count": invalid_rows, "invalid_weight_count": invalid_weights,
        "order_error_count": order_errors,
    }
    report_building = args.report.with_suffix(args.report.suffix + ".building")
    with report_building.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_building, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
