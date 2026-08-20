"""Create a reproducible, user-level Duolingo churn-analysis dataset.

The source CSV is never deleted or modified.  Large-file operations use pandas
chunks, and the final global sort uses small temporary user-range buckets.
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "p_recall",
    "timestamp",
    "delta",
    "user_id",
    "learning_language",
    "ui_language",
    "lexeme_id",
    "lexeme_string",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
]

FINAL_COLUMNS = [
    "p_recall",
    "timestamp",
    "delta",
    "user_id",
    "learning_language",
    "ui_language",
    "lexeme_id",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Duolingo 로그를 7일 이상 관찰 사용자 단위로 축소합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "data" / "learning_traces.13m.csv",
        help="원본 CSV 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "duolingo_churn_dataset.csv",
        help="최종 CSV 경로",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-observation-days", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-size-mib",
        type=float,
        default=100.0,
        help="표본 추출이 필요할 때 목표 크기(MiB)",
    )
    parser.add_argument(
        "--max-size-mib",
        type=float,
        default=200.0,
        help="최종 파일 허용 상한(MiB)",
    )
    parser.add_argument("--buckets", type=int, default=64)
    return parser.parse_args()


def format_bytes(size: int | float) -> str:
    return f"{int(size):,} bytes ({size / 1_000_000:.2f} MB, {size / 2**20:.2f} MiB)"


def utc_text(timestamp: int | float) -> str:
    return pd.to_datetime(timestamp, unit="s", utc=True).isoformat()


def progress(pass_name: str, chunk_number: int, rows: int, started: float) -> None:
    print(
        f"  {pass_name} chunk {chunk_number:>2}: {rows:,} rows "
        f"({time.time() - started:.1f}s)",
        flush=True,
    )


def inspect_columns(input_path: Path) -> list[str]:
    columns = list(pd.read_csv(input_path, nrows=0).columns)
    if columns != EXPECTED_COLUMNS:
        expected = set(EXPECTED_COLUMNS)
        actual = set(columns)
        raise ValueError(
            "원본 컬럼이 예상과 다릅니다.\n"
            f"  실제 순서: {columns}\n"
            f"  추가 컬럼: {sorted(actual - expected)}\n"
            f"  누락 컬럼: {sorted(expected - actual)}"
        )
    return columns


def aggregate_users(
    input_path: Path, chunksize: int
) -> tuple[pd.DataFrame, int, int, int]:
    summaries: list[pd.DataFrame] = []
    total_rows = 0
    source_min_timestamp: int | None = None
    source_max_timestamp: int | None = None
    started = time.time()

    reader = pd.read_csv(
        input_path,
        usecols=["user_id", "timestamp"],
        dtype={"user_id": "string"},
        chunksize=chunksize,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        valid = chunk.dropna(subset=["user_id", "timestamp"])
        if not valid.empty:
            local_min = int(valid["timestamp"].min())
            local_max = int(valid["timestamp"].max())
            source_min_timestamp = (
                local_min
                if source_min_timestamp is None
                else min(source_min_timestamp, local_min)
            )
            source_max_timestamp = (
                local_max
                if source_max_timestamp is None
                else max(source_max_timestamp, local_max)
            )
            grouped = (
                valid.groupby("user_id", sort=False, observed=True)["timestamp"]
                .agg(rows="size", first_timestamp="min", last_timestamp="max")
                .reset_index()
            )
            summaries.append(grouped)
        progress("사용자 집계", chunk_number, total_rows, started)

    if not summaries or source_min_timestamp is None or source_max_timestamp is None:
        raise ValueError("유효한 user_id/timestamp 행이 없습니다.")

    combined = pd.concat(summaries, ignore_index=True)
    users = combined.groupby("user_id", sort=False, observed=True).agg(
        rows=("rows", "sum"),
        first_timestamp=("first_timestamp", "min"),
        last_timestamp=("last_timestamp", "max"),
    )
    users["rows"] = users["rows"].astype("int64")
    users["first_timestamp"] = users["first_timestamp"].astype("int64")
    users["last_timestamp"] = users["last_timestamp"].astype("int64")
    users["observation_days"] = (
        users["last_timestamp"] - users["first_timestamp"]
    ) / 86_400.0
    return users, total_rows, source_min_timestamp, source_max_timestamp


def add_counts(target: dict[str, int], values: pd.Series) -> None:
    for key, value in values.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def measure_candidates(
    input_path: Path,
    candidate_users: set[str],
    chunksize: int,
) -> dict[str, object]:
    candidate_rows = 0
    payload_bytes = 0
    language_rows: dict[str, int] = {}
    ui_rows: dict[str, int] = {}
    language_users: dict[str, set[str]] = defaultdict(set)
    ui_users: dict[str, set[str]] = defaultdict(set)
    started = time.time()
    scanned_rows = 0

    reader = pd.read_csv(input_path, usecols=FINAL_COLUMNS, chunksize=chunksize)
    for chunk_number, chunk in enumerate(reader, start=1):
        scanned_rows += len(chunk)
        selected = chunk[chunk["user_id"].isin(candidate_users)]
        if not selected.empty:
            candidate_rows += len(selected)
            encoded = selected.to_csv(
                index=False, header=False, lineterminator="\n"
            ).encode("utf-8")
            payload_bytes += len(encoded)
            add_counts(language_rows, selected["learning_language"].value_counts())
            add_counts(ui_rows, selected["ui_language"].value_counts())

            language_pairs = selected[
                ["learning_language", "user_id"]
            ].drop_duplicates()
            for language, group in language_pairs.groupby(
                "learning_language", sort=False, observed=True
            ):
                language_users[str(language)].update(group["user_id"].astype(str))

            ui_pairs = selected[["ui_language", "user_id"]].drop_duplicates()
            for language, group in ui_pairs.groupby(
                "ui_language", sort=False, observed=True
            ):
                ui_users[str(language)].update(group["user_id"].astype(str))
        progress("후보 측정", chunk_number, scanned_rows, started)

    return {
        "rows": candidate_rows,
        "payload_bytes": payload_bytes,
        "language_rows": dict(sorted(language_rows.items())),
        "ui_rows": dict(sorted(ui_rows.items())),
        "language_users": {
            key: len(value) for key, value in sorted(language_users.items())
        },
        "ui_users": {key: len(value) for key, value in sorted(ui_users.items())},
    }


def print_distribution(
    title: str,
    user_counts: dict[str, int],
    row_counts: dict[str, int],
    total_rows: int,
) -> None:
    print(f"  {title} (고유 사용자 / rows):")
    for key in sorted(set(user_counts) | set(row_counts)):
        rows = row_counts.get(key, 0)
        ratio = rows / total_rows * 100 if total_rows else 0.0
        print(
            f"    {key}: {user_counts.get(key, 0):,}명 / "
            f"{rows:,} rows ({ratio:.2f}%)"
        )


def select_users(
    candidate_stats: pd.DataFrame,
    average_row_bytes: float,
    header_bytes: int,
    target_bytes: int,
    seed: int,
) -> tuple[list[str], int, float, str]:
    all_users = sorted(candidate_stats.index.astype(str).tolist())
    all_rows = int(candidate_stats["rows"].sum())
    all_estimated_bytes = header_bytes + all_rows * average_row_bytes

    if all_estimated_bytes <= target_bytes:
        return all_users, all_rows, all_estimated_bytes, (
            "7일 후보 전체의 예상 크기가 목표 이하이므로 전원을 유지"
        )

    shuffled = all_users.copy()
    random.Random(seed).shuffle(shuffled)
    selected: list[str] = []
    selected_rows = 0
    for user_id in shuffled:
        selected.append(user_id)
        selected_rows += int(candidate_stats.at[user_id, "rows"])
        if header_bytes + selected_rows * average_row_bytes >= target_bytes:
            break

    estimated_bytes = header_bytes + selected_rows * average_row_bytes
    reason = (
        f"7일 후보 전체 예상 크기({format_bytes(all_estimated_bytes)})가 목표를 "
        f"초과하여 seed={seed}의 user_id 단위 무작위 순서에서 목표 크기까지 선택"
    )
    return selected, selected_rows, estimated_bytes, reason


def extract_to_buckets(
    input_path: Path,
    selected_users: list[str],
    chunksize: int,
    bucket_count: int,
    temp_dir: Path,
) -> list[Path]:
    sorted_users = sorted(selected_users)
    actual_bucket_count = max(1, min(bucket_count, len(sorted_users)))
    user_to_bucket = {
        user_id: min(
            actual_bucket_count - 1,
            rank * actual_bucket_count // len(sorted_users),
        )
        for rank, user_id in enumerate(sorted_users)
    }
    bucket_paths = [temp_dir / f"bucket_{index:03d}.csv" for index in range(actual_bucket_count)]
    selected_set = set(selected_users)
    total_scanned = 0
    total_extracted = 0
    started = time.time()

    reader = pd.read_csv(input_path, usecols=FINAL_COLUMNS, chunksize=chunksize)
    for chunk_number, chunk in enumerate(reader, start=1):
        total_scanned += len(chunk)
        selected = chunk[chunk["user_id"].isin(selected_set)].copy()
        if not selected.empty:
            selected["_bucket"] = selected["user_id"].map(user_to_bucket)
            if selected["_bucket"].isna().any():
                raise RuntimeError("선택 사용자 버킷 매핑에 실패했습니다.")
            selected["_bucket"] = selected["_bucket"].astype("int16")
            total_extracted += len(selected)
            for bucket, group in selected.groupby("_bucket", sort=False, observed=True):
                group[FINAL_COLUMNS].to_csv(
                    bucket_paths[int(bucket)],
                    mode="a",
                    header=False,
                    index=False,
                    lineterminator="\n",
                )
        progress("최종 추출", chunk_number, total_scanned, started)

    print(f"  추출 rows: {total_extracted:,}")
    return bucket_paths


def sort_buckets_to_output(bucket_paths: list[Path], temporary_output: Path) -> None:
    temporary_output.unlink(missing_ok=True)
    wrote_header = False
    total_written = 0
    for index, bucket_path in enumerate(bucket_paths, start=1):
        if not bucket_path.exists() or bucket_path.stat().st_size == 0:
            continue
        bucket = pd.read_csv(
            bucket_path,
            names=FINAL_COLUMNS,
            header=None,
            dtype=str,
            keep_default_na=False,
        )
        bucket["_timestamp_sort"] = pd.to_numeric(
            bucket["timestamp"], errors="raise"
        ).astype("int64")
        bucket.sort_values(
            ["user_id", "_timestamp_sort"],
            kind="mergesort",
            inplace=True,
        )
        bucket.drop(columns="_timestamp_sort", inplace=True)
        bucket.to_csv(
            temporary_output,
            mode="a",
            header=not wrote_header,
            index=False,
            lineterminator="\n",
        )
        wrote_header = True
        total_written += len(bucket)
        print(
            f"  정렬 버킷 {index:>2}/{len(bucket_paths)}: "
            f"누적 {total_written:,} rows",
            flush=True,
        )
    if not wrote_header:
        raise RuntimeError("최종 출력에 기록할 행이 없습니다.")


def validate_output(
    output_path: Path,
    selected_users: list[str],
    expected_user_stats: pd.DataFrame,
    chunksize: int,
    max_bytes: int,
) -> dict[str, object]:
    actual_columns = list(pd.read_csv(output_path, nrows=0).columns)
    if actual_columns != FINAL_COLUMNS:
        raise AssertionError(f"최종 컬럼 불일치: {actual_columns}")

    total_rows = 0
    missing = pd.Series(0, index=FINAL_COLUMNS, dtype="int64")
    p_recall_min = np.inf
    p_recall_max = -np.inf
    p_recall_invalid = 0
    history_invalid = 0
    session_invalid = 0
    sorted_ok = True
    duplicate_rows = 0
    previous_user: str | None = None
    previous_timestamp: int | None = None
    pending_duplicate_group: pd.DataFrame | None = None
    user_summaries: list[pd.DataFrame] = []
    started = time.time()

    reader = pd.read_csv(output_path, chunksize=chunksize, low_memory=False)
    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")

        users = chunk["user_id"].astype(str).to_numpy()
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy()
        if len(chunk):
            if previous_user is not None:
                first_user = users[0]
                first_timestamp = timestamps[0]
                if first_user < previous_user or (
                    first_user == previous_user
                    and first_timestamp < previous_timestamp  # type: ignore[operator]
                ):
                    sorted_ok = False
            if len(chunk) > 1:
                bad_order = (users[1:] < users[:-1]) | (
                    (users[1:] == users[:-1])
                    & (timestamps[1:] < timestamps[:-1])
                )
                if bool(np.any(bad_order)):
                    sorted_ok = False
            previous_user = users[-1]
            previous_timestamp = int(timestamps[-1])

        p_recall = pd.to_numeric(chunk["p_recall"], errors="coerce")
        valid_p = p_recall.dropna()
        if not valid_p.empty:
            p_recall_min = min(p_recall_min, float(valid_p.min()))
            p_recall_max = max(p_recall_max, float(valid_p.max()))
        p_recall_invalid += int(((p_recall < 0) | (p_recall > 1)).sum())

        history_seen = pd.to_numeric(chunk["history_seen"], errors="coerce")
        history_correct = pd.to_numeric(chunk["history_correct"], errors="coerce")
        session_seen = pd.to_numeric(chunk["session_seen"], errors="coerce")
        session_correct = pd.to_numeric(chunk["session_correct"], errors="coerce")
        history_invalid += int((history_correct > history_seen).sum())
        session_invalid += int((session_correct > session_seen).sum())

        grouped = (
            chunk.assign(_timestamp_numeric=timestamps)
            .groupby("user_id", sort=False, observed=True)["_timestamp_numeric"]
            .agg(rows="size", first_timestamp="min", last_timestamp="max")
            .reset_index()
        )
        user_summaries.append(grouped)

        duplicate_work = (
            chunk
            if pending_duplicate_group is None
            else pd.concat([pending_duplicate_group, chunk], ignore_index=True)
        )
        last_user = duplicate_work.iloc[-1]["user_id"]
        last_timestamp = duplicate_work.iloc[-1]["timestamp"]
        pending_mask = (
            (duplicate_work["user_id"] == last_user)
            & (duplicate_work["timestamp"] == last_timestamp)
        )
        complete_groups = duplicate_work.loc[~pending_mask]
        duplicate_rows += int(complete_groups.duplicated().sum())
        pending_duplicate_group = duplicate_work.loc[pending_mask].copy()

        progress("검증", chunk_number, total_rows, started)

    if pending_duplicate_group is not None:
        duplicate_rows += int(pending_duplicate_group.duplicated().sum())

    combined = pd.concat(user_summaries, ignore_index=True)
    final_users = combined.groupby("user_id", sort=False, observed=True).agg(
        rows=("rows", "sum"),
        first_timestamp=("first_timestamp", "min"),
        last_timestamp=("last_timestamp", "max"),
    )
    final_users.index = final_users.index.astype(str)
    final_users["observation_days"] = (
        final_users["last_timestamp"] - final_users["first_timestamp"]
    ) / 86_400.0

    expected = expected_user_stats.loc[selected_users, "rows"].copy()
    expected.index = expected.index.astype(str)
    actual = final_users["rows"].reindex(expected.index)
    preserved_user_rows = bool(
        actual.notna().all()
        and len(final_users) == len(expected)
        and np.array_equal(actual.astype("int64").to_numpy(), expected.astype("int64").to_numpy())
    )

    result = {
        "file_bytes": output_path.stat().st_size,
        "rows": total_rows,
        "users": len(final_users),
        "min_timestamp": int(final_users["first_timestamp"].min()),
        "max_timestamp": int(final_users["last_timestamp"].max()),
        "observation_mean": float(final_users["observation_days"].mean()),
        "observation_median": float(final_users["observation_days"].median()),
        "duplicate_rows": duplicate_rows,
        "missing": {key: int(value) for key, value in missing.items()},
        "p_recall_min": float(p_recall_min),
        "p_recall_max": float(p_recall_max),
        "p_recall_invalid": p_recall_invalid,
        "history_invalid": history_invalid,
        "session_invalid": session_invalid,
        "sorted": sorted_ok,
        "preserved_user_rows": preserved_user_rows,
    }

    failures: list[str] = []
    if result["file_bytes"] > max_bytes:
        failures.append("최종 파일 크기가 상한을 초과함")
    if not sorted_ok:
        failures.append("user_id/timestamp 정렬 위반")
    if not preserved_user_rows:
        failures.append("선택 사용자의 원본 row 수 불일치")
    if duplicate_rows:
        failures.append(f"중복 row {duplicate_rows:,}건")
    if any(result["missing"].values()):  # type: ignore[union-attr]
        failures.append("주요 컬럼 결측치 존재")
    if p_recall_invalid:
        failures.append(f"p_recall 범위 위반 {p_recall_invalid:,}건")
    if history_invalid:
        failures.append(f"history_correct > history_seen {history_invalid:,}건")
    if session_invalid:
        failures.append(f"session_correct > session_seen {session_invalid:,}건")
    if failures:
        raise AssertionError("최종 검증 실패:\n- " + "\n- ".join(failures))

    return result


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    temporary_output = output_path.with_name(output_path.name + ".part")
    target_bytes = int(args.target_size_mib * 2**20)
    max_bytes = int(args.max_size_mib * 2**20)

    if not input_path.is_file():
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {input_path}")
    if input_path == output_path:
        raise ValueError("원본과 최종 출력 경로는 달라야 합니다.")
    if args.chunksize <= 0 or args.buckets <= 0:
        raise ValueError("chunksize와 buckets는 양수여야 합니다.")
    if args.min_observation_days < 0:
        raise ValueError("최소 관찰 기간은 0 이상이어야 합니다.")
    if target_bytes <= 0 or max_bytes < target_bytes:
        raise ValueError("파일 크기 목표/상한 설정이 잘못되었습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = input_path.stat().st_size

    print("[1/6] 원본 데이터 확인 중...", flush=True)
    columns = inspect_columns(input_path)
    print(f"  원본: {input_path}")
    print(f"  원본 크기: {format_bytes(source_bytes)}")
    print(f"  실제 컬럼: {columns}")
    print("  컬럼 차이: 없음")
    print("  최종 제외 컬럼: lexeme_string")

    print("\n[2/6] 사용자별 통계 계산 중...", flush=True)
    users, source_rows, source_min_timestamp, source_max_timestamp = aggregate_users(
        input_path, args.chunksize
    )
    candidate_stats = users[users["observation_days"] >= args.min_observation_days].copy()
    if candidate_stats.empty:
        raise ValueError(
            f"관찰 기간 {args.min_observation_days:g}일 이상 사용자가 없습니다."
        )
    candidate_users = set(candidate_stats.index.astype(str))

    print("\n[3/6] 7일 이상 후보 규모 확인 및 사용자 선정 중...", flush=True)
    measured = measure_candidates(input_path, candidate_users, args.chunksize)
    candidate_rows = int(measured["rows"])
    expected_candidate_rows = int(candidate_stats["rows"].sum())
    if candidate_rows != expected_candidate_rows:
        raise AssertionError(
            f"후보 row 집계 불일치: 1차={expected_candidate_rows:,}, "
            f"2차={candidate_rows:,}"
        )

    total_users = len(users)
    candidate_count = len(candidate_stats)
    print("\n  [7일 이상 후보 통계]")
    print(f"  전체 사용자 수: {total_users:,}")
    print(f"  7일 이상 사용자 수: {candidate_count:,}")
    print(f"  전체 rows: {source_rows:,}")
    print(f"  7일 이상 사용자 rows: {candidate_rows:,}")
    print(f"  남는 사용자 비율: {candidate_count / total_users * 100:.2f}%")
    print(f"  남는 row 비율: {candidate_rows / source_rows * 100:.2f}%")
    print(f"  사용자당 평균 rows: {candidate_stats['rows'].mean():,.2f}")
    print(f"  사용자당 중앙 rows: {candidate_stats['rows'].median():,.2f}")
    print(f"  평균 관찰 기간: {candidate_stats['observation_days'].mean():.4f}일")
    print(f"  중앙 관찰 기간: {candidate_stats['observation_days'].median():.4f}일")
    print(f"  최소 관찰 기간: {candidate_stats['observation_days'].min():.4f}일")
    print(f"  최대 관찰 기간: {candidate_stats['observation_days'].max():.4f}일")
    print_distribution(
        "learning_language 분포",
        measured["language_users"],  # type: ignore[arg-type]
        measured["language_rows"],  # type: ignore[arg-type]
        candidate_rows,
    )
    print_distribution(
        "ui_language 분포",
        measured["ui_users"],  # type: ignore[arg-type]
        measured["ui_rows"],  # type: ignore[arg-type]
        candidate_rows,
    )

    header_bytes = len((",".join(FINAL_COLUMNS) + "\n").encode("utf-8"))
    average_row_bytes = int(measured["payload_bytes"]) / candidate_rows
    average_user_bytes = int(measured["payload_bytes"]) / candidate_count
    selected_users, expected_selected_rows, estimated_bytes, reason = select_users(
        candidate_stats,
        average_row_bytes,
        header_bytes,
        target_bytes,
        args.seed,
    )
    print("\n  [사용자 선정]")
    print(f"  후보 row당 평균 CSV 크기: {average_row_bytes:.2f} bytes")
    print(f"  후보 사용자당 평균 CSV 크기: {format_bytes(average_user_bytes)}")
    print(f"  목표 크기: {format_bytes(target_bytes)}")
    print(f"  random seed: {args.seed}")
    print(f"  결정 사용자 수: {len(selected_users):,}명")
    print(f"  예상 rows: {expected_selected_rows:,}")
    print(f"  예상 최종 크기: {format_bytes(estimated_bytes)}")
    print(f"  결정 이유: {reason}")

    temporary_output.unlink(missing_ok=True)
    try:
        print("\n[4/6] 선택 사용자의 전체 데이터 추출 중...", flush=True)
        with tempfile.TemporaryDirectory(
            prefix="duolingo_prepare_", dir=output_path.parent
        ) as temp_name:
            bucket_paths = extract_to_buckets(
                input_path,
                selected_users,
                args.chunksize,
                args.buckets,
                Path(temp_name),
            )

            print("\n[5/6] user_id, timestamp 순 정렬 및 CSV 저장 중...", flush=True)
            sort_buckets_to_output(bucket_paths, temporary_output)

        print("\n[6/6] 데이터 검증 중...", flush=True)
        validation = validate_output(
            temporary_output,
            selected_users,
            candidate_stats,
            args.chunksize,
            max_bytes,
        )
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    print("\n  [원본과 최종 데이터 비교]")
    print(f"  원본 파일 크기: {format_bytes(source_bytes)}")
    print(f"  최종 파일 크기: {format_bytes(validation['file_bytes'])}")
    print(f"  원본 rows: {source_rows:,}")
    print(f"  최종 rows: {validation['rows']:,}")
    print(f"  원본 사용자 수: {total_users:,}")
    print(f"  최종 사용자 수: {validation['users']:,}")
    print(f"  원본 최초 timestamp: {source_min_timestamp} ({utc_text(source_min_timestamp)})")
    print(f"  원본 마지막 timestamp: {source_max_timestamp} ({utc_text(source_max_timestamp)})")
    print(
        f"  최종 최초 timestamp: {validation['min_timestamp']} "
        f"({utc_text(validation['min_timestamp'])})"
    )
    print(
        f"  최종 마지막 timestamp: {validation['max_timestamp']} "
        f"({utc_text(validation['max_timestamp'])})"
    )
    print(f"  최종 사용자 평균 관찰 기간: {validation['observation_mean']:.4f}일")
    print(f"  최종 사용자 중앙 관찰 기간: {validation['observation_median']:.4f}일")

    print("\n  [검증 결과]")
    print(f"  중복 row: {validation['duplicate_rows']:,}건 (PASS)")
    print(f"  주요 컬럼 결측치: {validation['missing']} (PASS)")
    print(
        f"  p_recall 범위: {validation['p_recall_min']:.6g} ~ "
        f"{validation['p_recall_max']:.6g}, 위반 {validation['p_recall_invalid']:,}건 (PASS)"
    )
    print(
        f"  history_correct > history_seen: "
        f"{validation['history_invalid']:,}건 (PASS)"
    )
    print(
        f"  session_correct > session_seen: "
        f"{validation['session_invalid']:,}건 (PASS)"
    )
    print(f"  user_id, timestamp 정렬: {validation['sorted']} (PASS)")
    print(f"  선택 사용자 전체 row 보존: {validation['preserved_user_rows']} (PASS)")
    print("  Churn/파생 label 생성: 없음 (PASS)")
    print(f"\n완료: {output_path}")
    print("원본 CSV는 수정하거나 삭제하지 않았습니다.")


if __name__ == "__main__":
    main()
