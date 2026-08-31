"""원본/전처리 데이터 로드와 학습 입력 스키마 검증."""

import os

import pandas as pd
import pymysql
from dotenv import load_dotenv

from src.config import PROCESSED_DIR, RAW_DIR
from src.utils.constants import TARGET_COLUMN


def load_raw_train():
    return pd.read_csv(RAW_DIR / "train.csv")


def load_raw_test():
    return pd.read_csv(RAW_DIR / "test.csv")


def load_processed_train():
    return pd.read_csv(PROCESSED_DIR / "train_processed.csv")


def load_processed_test():
    return pd.read_csv(PROCESSED_DIR / "test_processed.csv")


# ==============================================================================
# DB(employee_attrition_processed) 기반 전처리 데이터 로드
# ==============================================================================
#
# DB 컬럼(snake_case) -> CSV 기반 전처리 결과 컬럼명 매핑.
#
# 학습/추론 코드(prepare_dataloaders, prepare_test_data 등)가
# "Attrition", "Job Role_Finance" 같은 CSV 컬럼명 규칙을 그대로 사용하므로,
# DB에서 읽어온 데이터도 동일한 이름으로 되돌려 기존 코드와 호환되게 한다.
#
# 참고: employee_attrition_processed.marital_status 컬럼은 실제로는 값이
# 채워지지 않는(항상 NULL) 컬럼이어서 DB에서 제거되었다. Marital Status는
# 원-핫 인코딩되어 marital_status_married / marital_status_single 두 컬럼으로만
# 존재한다.
DB_PROCESSED_COLUMN_MAPPING = {
    "age": "Age",
    "gender": "Gender",
    "years_at_company": "Years at Company",
    "monthly_income": "Monthly Income",
    "work_life_balance": "Work-Life Balance",
    "job_satisfaction": "Job Satisfaction",
    "performance_rating": "Performance Rating",
    "number_of_promotions": "Number of Promotions",
    "overtime": "Overtime",
    "distance_from_home": "Distance from Home",
    "education_level": "Education Level",
    "number_of_dependents": "Number of Dependents",
    "job_level": "Job Level",
    "company_size": "Company Size",
    "company_tenure": "Company Tenure",
    "remote_work": "Remote Work",
    "leadership_opportunities": "Leadership Opportunities",
    "innovation_opportunities": "Innovation Opportunities",
    "company_reputation": "Company Reputation",
    "employee_recognition": "Employee Recognition",
    "attrition": TARGET_COLUMN,  # "Attrition"
    "industry_experience_gap": "Industry Experience Gap",
    "promotion_rate": "Promotion Rate",
    "job_role_finance": "Job Role_Finance",
    "job_role_healthcare": "Job Role_Healthcare",
    "job_role_media": "Job Role_Media",
    "job_role_technology": "Job Role_Technology",
    "marital_status_married": "Marital Status_Married",
    "marital_status_single": "Marital Status_Single",
}

# employee_attrition_processed 테이블에는 있지만 피처로 사용하지 않는 메타 컬럼
DB_META_COLUMNS = ("processed_id", "employee_id", "type")


def _get_db_connection():
    """환경 변수(.env)를 이용해 MySQL(TiDB) 연결을 생성한다."""

    load_dotenv()

    required_variables = ("DB_HOST", "DB_USERNAME", "DB_PASSWORD", "DB_DATABASE")
    missing_variables = [name for name in required_variables if not os.getenv(name)]
    if missing_variables:
        raise ValueError("DB 환경 변수가 설정되지 않았습니다: " + ", ".join(missing_variables))

    return pymysql.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_DATABASE"],
        port=int(str(os.getenv("DB_PORT", "3306")).strip()),
        charset="utf8mb4",
    )


def _load_processed_from_db(data_type: str) -> pd.DataFrame:
    """employee_attrition_processed 테이블에서 train/test 전처리 데이터를 조회한다.

    반환 컬럼명은 기존 CSV(train_processed.csv / test_processed.csv) 기반
    전처리 결과와 동일하게 맞춰서, 이 함수를 호출하는 학습/추론 코드를
    수정하지 않아도 되도록 한다.
    """

    if data_type not in {"train", "test"}:
        raise ValueError("data_type은 'train' 또는 'test'만 가능합니다.")

    connection = _get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM employee_attrition_processed WHERE type = %s",
                (data_type,),
            )
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
    finally:
        connection.close()

    if not rows:
        raise ValueError(
            f"employee_attrition_processed 테이블에 type='{data_type}' 데이터가 없습니다."
        )

    df = pd.DataFrame(rows, columns=columns)

    # --------------------------------------------------------------------------
    # 컬럼 스키마 검증
    #
    # DB 테이블 구조가 바뀌면(컬럼 추가/삭제) 조용히 잘못된 데이터로
    # 학습하지 않도록 여기서 명시적으로 검증한다.
    # --------------------------------------------------------------------------

    actual_columns = set(df.columns)
    expected_columns = set(DB_PROCESSED_COLUMN_MAPPING) | set(DB_META_COLUMNS)

    missing_columns = set(DB_PROCESSED_COLUMN_MAPPING) - actual_columns
    if missing_columns:
        raise ValueError(
            "employee_attrition_processed 테이블에 예상 컬럼이 없습니다: "
            + ", ".join(sorted(missing_columns))
        )

    unexpected_columns = actual_columns - expected_columns
    if unexpected_columns:
        raise ValueError(
            "employee_attrition_processed 테이블에 매핑되지 않은 컬럼이 있습니다: "
            + ", ".join(sorted(unexpected_columns))
            + " (src/data/loader.py의 DB_PROCESSED_COLUMN_MAPPING을 갱신하세요.)"
        )

    # --------------------------------------------------------------------------
    # 메타 컬럼 제거 및 컬럼명을 CSV 기반 전처리 결과와 동일하게 정리
    # --------------------------------------------------------------------------

    df = df.drop(columns=[c for c in DB_META_COLUMNS if c in df.columns])
    df = df.rename(columns=DB_PROCESSED_COLUMN_MAPPING)

    # --------------------------------------------------------------------------
    # 숫자형 변환
    #
    # pymysql은 DECIMAL 컬럼(예: promotion_rate)을 Decimal 객체로 반환하므로
    # 학습에 바로 사용할 수 있도록 전체 컬럼을 숫자형으로 변환한다.
    # --------------------------------------------------------------------------

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isnull().any().any():
        null_columns = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"DB에서 로드한 {data_type} 데이터에 결측치가 있습니다: {null_columns}"
        )

    return df.reset_index(drop=True)


def load_processed_train_from_db() -> pd.DataFrame:
    """employee_attrition_processed 테이블(type='train')에서 학습 데이터를 로드한다."""

    return _load_processed_from_db("train")


def load_processed_test_from_db() -> pd.DataFrame:
    """employee_attrition_processed 테이블(type='test')에서 테스트 데이터를 로드한다."""

    return _load_processed_from_db("test")


def load_test_model_results_from_db() -> pd.DataFrame:
    """``test_model_results``의 ML/DL 성능을 화면용 컬럼명으로 조회한다."""

    connection = _get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
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
                FROM test_model_results
                ORDER BY roc_auc_score DESC, f1_score DESC, id ASC
                """
            )
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
    finally:
        connection.close()

    if not rows:
        raise ValueError(
            "test_model_results 테이블이 비어 있습니다. "
            "먼저 `uv run python insert_database.py`를 실행하세요."
        )

    results = pd.DataFrame(rows, columns=columns)
    results = results.rename(
        columns={
            "accuracy_score": "accuracy",
            "precision_score": "precision",
            "recall_score": "recall",
            "f1_score": "f1",
            "roc_auc_score": "roc_auc",
            "average_precision_score": "average_precision",
        }
    )
    metric_columns = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    )
    for column in metric_columns:
        results[column] = pd.to_numeric(results[column], errors="raise").astype(float)
    for column in ("tn", "fp", "fn", "tp"):
        results[column] = pd.to_numeric(results[column], errors="raise").astype(int)

    return results


def split_processed_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """전처리 데이터를 수치형 피처와 이진 타깃으로 분리하고 검증한다."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {TARGET_COLUMN}")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError("전처리 타깃에 숫자로 변환할 수 없는 값이 있습니다.")

    unexpected_labels = set(target.unique()) - {0, 1}
    if unexpected_labels:
        labels = ", ".join(map(str, sorted(unexpected_labels)))
        raise ValueError(f"예상하지 못한 타깃 값입니다: {labels}")

    saved_index_columns = [column for column in data.columns if column.startswith("Unnamed:")]
    features = data.drop(columns=[TARGET_COLUMN, *saved_index_columns]).copy()
    bool_columns = features.select_dtypes(include="bool").columns
    features[bool_columns] = features[bool_columns].astype("int8")

    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError("전처리되지 않은 피처가 있습니다: " + ", ".join(non_numeric_columns))

    return features, target.astype("int8").rename(TARGET_COLUMN)


def load_processed_train_test_features(
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """전처리 학습·테스트 데이터를 피처/타깃으로 로드하고 스키마를 검증한다."""

    train_features, train_target = split_processed_features_target(load_processed_train())
    test_features, test_target = split_processed_features_target(load_processed_test())

    if list(train_features.columns) != list(test_features.columns):
        missing_from_test = sorted(set(train_features.columns) - set(test_features.columns))
        extra_in_test = sorted(set(test_features.columns) - set(train_features.columns))
        raise ValueError(
            "학습/테스트 피처 구성이 다릅니다. "
            f"테스트에 없음={missing_from_test}, 테스트에만 있음={extra_in_test}"
        )

    return train_features, train_target, test_features, test_target
