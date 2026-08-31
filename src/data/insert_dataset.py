"""ML 리더보드와 DL 테스트 성능을 MySQL ``test_model_results``에 저장한다."""

import os

import pandas as pd
import pymysql
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from src.data.loader import load_processed_test
from src.data.prediction import (
    INSERT_PREDICTION_SQL,
    create_employee_predictions,
    prediction_rows,
)
from src.models.dl.predict import load_mlp_prediction_model
from src.utils.constants import TARGET_COLUMN
from src.utils.paths import (
    DL_METRICS_PATH,
    ML_LEADERBOARD_PATH,
    MLP_MODEL_PATH,
    PROJECT_ROOT,
    project_relative_path,
)

INSERT_TEST_MODEL_RESULT_SQL = """
    INSERT INTO test_model_results (
        model,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        tn,
        fp,
        fn,
        tp,
        artifact_path
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)
CONFUSION_COLUMNS = ("tn", "fp", "fn", "tp")


def get_db_connection():
    """환경 변수에 설정된 MySQL 연결을 생성한다."""

    load_dotenv()
    required_variables = ("DB_HOST", "DB_USERNAME", "DB_PASSWORD", "DB_DATABASE")
    missing_variables = [name for name in required_variables if not os.getenv(name)]
    if missing_variables:
        raise ValueError(
            "DB 환경 변수가 설정되지 않았습니다: " + ", ".join(missing_variables)
        )

    return pymysql.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_DATABASE"],
        port=int(os.getenv("DB_PORT", "3306").strip()),
        charset="utf8mb4",
    )


def insert_employee_attrition_predictions() -> int:
    """MLP로 계산한 test.csv 직원별 퇴사 확률을 DB에 저장한다.

    현재 ``main()``에서는 호출을 주석 처리하여 비활성화한 상태다.
    """

    predictions = create_employee_predictions()
    rows = prediction_rows(predictions)

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.executemany(INSERT_PREDICTION_SQL, rows)
        connection.commit()
        return len(rows)
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def _load_single_report(report_path) -> dict[str, object]:
    if not report_path.exists():
        raise FileNotFoundError(f"테스트 성능 리포트가 없습니다: {report_path}")

    report = pd.read_csv(report_path)
    if len(report) != 1:
        raise ValueError(f"테스트 성능 리포트는 정확히 한 행이어야 합니다: {report_path}")

    required = {"model", *METRIC_COLUMNS}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(
            f"{report_path.name}에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}"
        )
    return report.iloc[0].to_dict()


def _mlp_confusion_counts() -> dict[str, int]:
    """저장된 MLP 임계값으로 test 데이터의 혼동행렬을 계산한다."""

    test_data = load_processed_test()
    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(f"테스트 데이터에 타깃 컬럼이 없습니다: {TARGET_COLUMN}")

    model = load_mlp_prediction_model()
    feature_names = model.feature_names_in_.tolist()
    missing_features = [name for name in feature_names if name not in test_data.columns]
    if missing_features:
        raise ValueError("MLP 테스트 피처가 없습니다: " + ", ".join(missing_features))

    target = pd.to_numeric(test_data[TARGET_COLUMN], errors="raise").astype(int)
    predictions = model.predict(test_data.reindex(columns=feature_names))
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def load_test_model_result_rows() -> list[tuple[object, ...]]:
    """ML 리더보드 전체와 MLP 테스트 결과를 DB 행으로 변환한다."""

    if not ML_LEADERBOARD_PATH.exists():
        raise FileNotFoundError(f"ML 리더보드가 없습니다: {ML_LEADERBOARD_PATH}")

    leaderboard = pd.read_csv(ML_LEADERBOARD_PATH)
    required_ml_columns = {
        "model",
        "artifact_path",
        *METRIC_COLUMNS,
        *CONFUSION_COLUMNS,
    }
    missing_ml_columns = required_ml_columns.difference(leaderboard.columns)
    if missing_ml_columns:
        raise ValueError(
            "ML 리더보드에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_ml_columns))
        )

    rows: list[tuple[object, ...]] = []
    for result in leaderboard.to_dict(orient="records"):
        artifact_reference = str(result["artifact_path"]).replace("\\", "/")
        artifact_path = PROJECT_ROOT / artifact_reference
        if not artifact_path.exists():
            raise FileNotFoundError(f"ML 모델 아티팩트가 없습니다: {artifact_path}")
        rows.append(
            (
                str(result["model"]),
                *(round(float(result[name]), 5) for name in METRIC_COLUMNS),
                *(int(result[name]) for name in CONFUSION_COLUMNS),
                artifact_reference,
            )
        )

    mlp_result = _load_single_report(DL_METRICS_PATH)
    mlp_result.update(_mlp_confusion_counts())
    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(f"MLP 모델 아티팩트가 없습니다: {MLP_MODEL_PATH}")
    rows.append(
        (
            str(mlp_result["model"]),
            *(round(float(mlp_result[name]), 5) for name in METRIC_COLUMNS),
            *(int(mlp_result[name]) for name in CONFUSION_COLUMNS),
            project_relative_path(MLP_MODEL_PATH),
        )
    )

    return rows


def insert_test_model_results() -> int:
    """기존 결과를 비우고 최신 ML 리더보드와 MLP 성능을 저장한다."""

    rows = load_test_model_result_rows()

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM test_model_results")
        cursor.executemany(INSERT_TEST_MODEL_RESULT_SQL, rows)
        connection.commit()
        return len(rows)
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def main() -> None:
    # 기존 직원별 퇴사 예측 INSERT 기능은 보존하되 실행만 비활성화합니다.
    # inserted_count = insert_employee_attrition_predictions()
    # print(f"퇴사 예측 INSERT 완료: {inserted_count}건")

    inserted_count = insert_test_model_results()
    print(f"ML/DL 테스트 성능 INSERT 완료: {inserted_count}건")


if __name__ == "__main__":
    main()
