#!/usr/bin/env python3
"""Build atomic, restartable datasets for KT4 paying users.

The raw output is ordered lexicographically by ``user_id`` and then by timestamp.
Each worker writes an independently validated chunk. Completed chunks are reused
after an interrupted run, and final files replace their destinations only after
all chunks have completed successfully.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KT4_DIR = PROJECT_ROOT / "KT4"
DATA_DIR = PROJECT_ROOT / "data"
SUMMARY_PATH = DATA_DIR / "kt4_pass_expiry_repurchase_analysis.csv"
RAW_OUTPUT = DATA_DIR / "ednet_payment_users_full.csv"
FEATURE_OUTPUT = DATA_DIR / "churn_modeling_features.csv"
CHUNK_DIR = DATA_DIR / ".consolidated_payment_chunks"

OBS_WINDOW_DAYS = 14
MS_PER_DAY = 86_400_000.0
EXPECTED_USERS = 23_789
EXPECTED_EVENTS = 72_875_338
EXPECTED_CHURN = {0: 2_174, 1: 21_615}

SOURCE_FIELDS = [
    "timestamp", "action_type", "item_id", "cursor_time", "source",
    "user_answer", "platform",
]
RAW_FIELDS = ["user_id", *SOURCE_FIELDS]
FEATURE_FIELDS = [
    "user_id", "first_pay_ts", "first_pay_item", "pre_pay_events",
    "pre_pay_active_days", "pre_pay_span_days", "pre_pay_recency_days",
    "obs_total_events", "obs_active_days", "obs_events_per_active_day",
    "obs_w1_events", "obs_w2_events", "obs_decay_ratio",
    "obs_activity_change_rate", "obs_last_active_day", "obs_recency_days",
    "obs_unique_items", "obs_unique_sources", "obs_unique_platforms",
    "obs_enter_count", "obs_respond_count", "obs_submit_count",
    "obs_response_with_answer_count", "obs_response_with_answer_rate",
    "obs_play_video_count", "obs_pause_video_count", "obs_play_audio_count",
    "obs_pause_audio_count", "obs_quit_count", "obs_pay_count",
    "obs_refund_count", "total_lifetime_events", "is_refund_churn",
    "is_non_renewal_churn", "is_churn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kt4-dir", type=Path, default=KT4_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    parser.add_argument("--feature-output", type=Path, default=FEATURE_OUTPUT)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Development-only number of paying users; disables fixed-count checks.",
    )
    parser.add_argument(
        "--discard-chunks", action="store_true",
        help="Remove restart chunks after final files are assembled.",
    )
    return parser.parse_args()


def as_int(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    return int(float(value)) if value else 0


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if not value:
        raise ValueError(f"Missing required value {field!r} for {row.get('user_id')}")
    return float(value)


def load_paying_users(summary_path: Path, limit: int | None) -> list[dict[str, Any]]:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("has_pay") == "1"]
    rows.sort(key=lambda row: row["user_id"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({
            "user_id": row["user_id"],
            "first_pay_ts": as_float(row, "first_pay_ts"),
            "first_pay_item": row.get("first_pay_item", ""),
            "last_pay_item": row.get("last_pay_item", ""),
            "pay_count": as_int(row, "pay_count"),
            "refund_count": as_int(row, "refund_count"),
            "has_refund": as_int(row, "has_refund"),
            "has_repurchase": as_int(row, "has_repurchase"),
            "total_events": as_int(row, "total_events"),
            "is_refund_churn": as_int(row, "is_refund_churn"),
            "is_non_renewal_churn": as_int(row, "is_non_renewal_churn"),
            "is_churn": as_int(row, "is_churn_overall"),
        })
    return result


def batch_fingerprint(batch: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in batch:
        digest.update(
            f"{row['user_id']}|{row['first_pay_ts']}|{row['total_events']}|"
            f"{row['is_churn']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def finite_timestamp(raw: str, user_id: str, row_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp for {user_id} at source row {row_number}: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite timestamp for {user_id} at source row {row_number}: {raw!r}"
        )
    return value


def initial_counters() -> dict[str, Any]:
    return {
        "pre_events": 0, "pre_dates": set(), "pre_first_ts": None,
        "pre_last_ts": None, "obs_events": 0, "obs_dates": set(),
        "obs_w1_events": 0, "obs_w2_events": 0, "obs_last_ts": None,
        "obs_items": set(), "obs_sources": set(), "obs_platforms": set(),
        "actions": {}, "response_with_answer": 0,
    }


def update_counters(
    counters: dict[str, Any], values: list[str], timestamp: float, first_pay_ts: float
) -> None:
    if timestamp < first_pay_ts:
        counters["pre_events"] += 1
        counters["pre_dates"].add(int(timestamp // MS_PER_DAY))
        if counters["pre_first_ts"] is None:
            counters["pre_first_ts"] = timestamp
        counters["pre_last_ts"] = timestamp
        return
    obs_end = first_pay_ts + OBS_WINDOW_DAYS * MS_PER_DAY
    if timestamp > obs_end:
        return

    action = values[1].strip().casefold()
    counters["obs_events"] += 1
    counters["obs_dates"].add(int(timestamp // MS_PER_DAY))
    counters["obs_last_ts"] = timestamp
    if timestamp < first_pay_ts + 7 * MS_PER_DAY:
        counters["obs_w1_events"] += 1
    else:
        counters["obs_w2_events"] += 1
    if values[2].strip():
        counters["obs_items"].add(values[2].strip())
    if values[4].strip():
        counters["obs_sources"].add(values[4].strip())
    if values[6].strip():
        counters["obs_platforms"].add(values[6].strip())
    counters["actions"][action] = counters["actions"].get(action, 0) + 1
    if action == "respond" and values[5].strip():
        counters["response_with_answer"] += 1


def rounded(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def make_feature(row: dict[str, Any], counters: dict[str, Any]) -> dict[str, Any]:
    first_pay_ts = row["first_pay_ts"]
    pre_first_ts = counters["pre_first_ts"]
    pre_last_ts = counters["pre_last_ts"]
    obs_last_ts = counters["obs_last_ts"]
    w1 = counters["obs_w1_events"]
    w2 = counters["obs_w2_events"]
    active_days = len(counters["obs_dates"])
    respond_count = counters["actions"].get("respond", 0)
    pre_span = 0.0 if pre_first_ts is None else (pre_last_ts - pre_first_ts) / MS_PER_DAY
    pre_recency = 0.0 if pre_last_ts is None else (first_pay_ts - pre_last_ts) / MS_PER_DAY
    last_active_day = 0.0 if obs_last_ts is None else (obs_last_ts - first_pay_ts) / MS_PER_DAY
    recency = max(0.0, OBS_WINDOW_DAYS - last_active_day)
    action = counters["actions"]
    return {
        "user_id": row["user_id"], "first_pay_ts": rounded(first_pay_ts),
        "first_pay_item": row["first_pay_item"],
        "pre_pay_events": counters["pre_events"],
        "pre_pay_active_days": len(counters["pre_dates"]),
        "pre_pay_span_days": rounded(max(0.0, pre_span)),
        "pre_pay_recency_days": rounded(max(0.0, pre_recency)),
        "obs_total_events": counters["obs_events"], "obs_active_days": active_days,
        "obs_events_per_active_day": rounded(counters["obs_events"] / max(1, active_days)),
        "obs_w1_events": w1, "obs_w2_events": w2,
        "obs_decay_ratio": rounded(w2 / max(1, w1)),
        "obs_activity_change_rate": rounded((w2 - w1) / max(1, w1)),
        "obs_last_active_day": rounded(max(0.0, last_active_day)),
        "obs_recency_days": rounded(recency),
        "obs_unique_items": len(counters["obs_items"]),
        "obs_unique_sources": len(counters["obs_sources"]),
        "obs_unique_platforms": len(counters["obs_platforms"]),
        "obs_enter_count": action.get("enter", 0),
        "obs_respond_count": respond_count,
        "obs_submit_count": action.get("submit", 0),
        "obs_response_with_answer_count": counters["response_with_answer"],
        "obs_response_with_answer_rate": rounded(counters["response_with_answer"] / max(1, respond_count)),
        "obs_play_video_count": action.get("play_video", 0),
        "obs_pause_video_count": action.get("pause_video", 0),
        "obs_play_audio_count": action.get("play_audio", 0),
        "obs_pause_audio_count": action.get("pause_audio", 0),
        "obs_quit_count": action.get("quit", 0), "obs_pay_count": action.get("pay", 0),
        "obs_refund_count": action.get("refund", 0),
        "total_lifetime_events": row["total_events"],
        "is_refund_churn": row["is_refund_churn"],
        "is_non_renewal_churn": row["is_non_renewal_churn"], "is_churn": row["is_churn"],
    }


def process_batch(task: tuple[int, list[dict[str, Any]], str, str]) -> dict[str, Any]:
    batch_index, batch, kt4_dir_raw, chunk_dir_raw = task
    kt4_dir = Path(kt4_dir_raw)
    chunk_dir = Path(chunk_dir_raw)
    fingerprint = batch_fingerprint(batch)
    chunk_path = chunk_dir / f"chunk_{batch_index:05d}.csv"
    metadata_path = chunk_dir / f"chunk_{batch_index:05d}.json"
    if chunk_path.is_file() and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("user_count") == len(batch)
            and metadata.get("chunk_bytes") == chunk_path.stat().st_size
        ):
            return metadata

    chunk_partial = chunk_path.with_suffix(".csv.partial")
    metadata_partial = metadata_path.with_suffix(".json.partial")
    features: list[dict[str, Any]] = []
    batch_rows = 0
    with chunk_partial.open(
        "w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024
    ) as output_handle:
        writer = csv.writer(output_handle)
        for summary_row in batch:
            user_id = summary_row["user_id"]
            source_path = kt4_dir / f"{user_id}.csv"
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing KT4 source file: {source_path}")
            counters = initial_counters()
            user_rows: list[tuple[float, int, list[str]]] = []
            with source_path.open(
                "r", encoding="utf-8-sig", newline="", buffering=1024 * 1024
            ) as source_handle:
                reader = csv.reader(source_handle, strict=True)
                header = next(reader, None)
                if header is None:
                    raise ValueError(f"Empty source file: {source_path}")
                normalized = [name.strip().casefold() for name in header]
                missing = [field for field in SOURCE_FIELDS if field not in normalized]
                if missing:
                    raise ValueError(f"Missing columns {missing} in {source_path}")
                indexes = [normalized.index(field) for field in SOURCE_FIELDS]
                required_index = max(indexes)
                for row_number, source_row in enumerate(reader, start=2):
                    if len(source_row) <= required_index:
                        raise ValueError(
                            f"Short row in {source_path} at {row_number}: expected index "
                            f"{required_index}, got {len(source_row)} columns"
                        )
                    values = [source_row[index] for index in indexes]
                    timestamp = finite_timestamp(values[0], user_id, row_number)
                    user_rows.append((timestamp, row_number, values))
            if len(user_rows) != summary_row["total_events"]:
                raise ValueError(
                    f"Source count mismatch for {user_id}: summary={summary_row['total_events']}, "
                    f"source={len(user_rows)}"
                )
            user_rows.sort(key=lambda item: (item[0], item[1]))
            for timestamp, _row_number, values in user_rows:
                writer.writerow([user_id, *values])
                update_counters(counters, values, timestamp, summary_row["first_pay_ts"])
            batch_rows += len(user_rows)
            features.append(make_feature(summary_row, counters))
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(chunk_partial, chunk_path)
    metadata = {
        "batch_index": batch_index, "fingerprint": fingerprint,
        "user_count": len(batch), "row_count": batch_rows,
        "chunk_bytes": chunk_path.stat().st_size, "features": features,
    }
    with metadata_partial.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(metadata_partial, metadata_path)
    return metadata


def validate_summary_expectations(users: list[dict[str, Any]], limited: bool) -> None:
    if limited:
        return
    event_count = sum(row["total_events"] for row in users)
    churn = {label: sum(row["is_churn"] == label for row in users) for label in (0, 1)}
    errors = []
    if len(users) != EXPECTED_USERS:
        errors.append(f"paying users: expected {EXPECTED_USERS:,}, got {len(users):,}")
    if event_count != EXPECTED_EVENTS:
        errors.append(f"events: expected {EXPECTED_EVENTS:,}, got {event_count:,}")
    if churn != EXPECTED_CHURN:
        errors.append(f"churn distribution: expected {EXPECTED_CHURN}, got {churn}")
    if errors:
        raise ValueError("Summary expectation failure: " + "; ".join(errors))


def assemble_raw(output_path: Path, chunk_dir: Path, metadata: list[dict[str, Any]]) -> None:
    building = output_path.with_suffix(output_path.suffix + ".building")
    with building.open("wb", buffering=32 * 1024 * 1024) as destination:
        destination.write((",".join(RAW_FIELDS) + "\r\n").encode("utf-8"))
        for item in metadata:
            chunk_path = chunk_dir / f"chunk_{item['batch_index']:05d}.csv"
            with chunk_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=32 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(building, output_path)


def assemble_features(output_path: Path, metadata: list[dict[str, Any]]) -> None:
    building = output_path.with_suffix(output_path.suffix + ".building")
    with building.open("w", encoding="utf-8", newline="") as handle:
        # Cached restart metadata from an older schema can contain retired fields.
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in metadata:
            writer.writerows(item["features"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(building, output_path)


def format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / (1024 ** 3):.2f} GiB"
    return f"{size / (1024 ** 2):.2f} MiB"


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    kt4_dir = args.kt4_dir.resolve()
    summary_path = args.summary.resolve()
    raw_output = args.raw_output.resolve()
    feature_output = args.feature_output.resolve()
    chunk_dir = args.chunk_dir.resolve()
    if not kt4_dir.is_dir():
        raise FileNotFoundError(f"KT4 directory not found: {kt4_dir}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("--workers and --batch-size must be positive")

    users = load_paying_users(summary_path, args.limit)
    validate_summary_expectations(users, args.limit is not None)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    batches = [users[index:index + args.batch_size] for index in range(0, len(users), args.batch_size)]
    tasks = [(index, batch, str(kt4_dir), str(chunk_dir)) for index, batch in enumerate(batches)]
    print(
        f"Paying users: {len(users):,}; expected events: "
        f"{sum(row['total_events'] for row in users):,}; batches: {len(batches):,}; "
        f"workers: {args.workers}", flush=True,
    )

    metadata: list[dict[str, Any]] = []
    completed_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, item in enumerate(executor.map(process_batch, tasks), start=1):
            metadata.append(item)
            completed_rows += item["row_count"]
            print(
                f"Chunks {completed:,}/{len(tasks):,}; rows {completed_rows:,}; "
                f"elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True,
            )
    expected_rows = sum(row["total_events"] for row in users)
    if completed_rows != expected_rows:
        raise ValueError(f"Chunk row total mismatch: chunks={completed_rows:,}, summary={expected_rows:,}")
    if sum(item["user_count"] for item in metadata) != len(users):
        raise ValueError("Chunk user total mismatch")

    print("Assembling final raw and feature files atomically...", flush=True)
    assemble_raw(raw_output, chunk_dir, metadata)
    assemble_features(feature_output, metadata)
    if args.discard_chunks:
        shutil.rmtree(chunk_dir)
    print(f"Raw: {raw_output} ({format_size(raw_output.stat().st_size)}, {completed_rows:,} rows)")
    print(
        f"Features: {feature_output} ({format_size(feature_output.stat().st_size)}, "
        f"{len(users):,} rows x {len(FEATURE_FIELDS)} columns)"
    )
    print(f"Elapsed: {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
