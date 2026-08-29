"""퇴사 예측 결과를 MySQL에 저장하는 실행 스크립트."""

import os

import pymysql
from dotenv import load_dotenv

from src.data.prediction import (
    INSERT_PREDICTION_SQL,
    create_employee_predictions,
    prediction_rows,
)


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
    """test.csv 전체 직원의 퇴사 확률을 DB에 저장한다."""

    predictions = create_employee_predictions()
    rows = prediction_rows(predictions)

    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM employee_attrition_prediction")
        cursor.executemany(
            """
            INSERT INTO employee_attrition_prediction (
                employee_id,
                prediction
            )
            VALUES (%s, %s)
            """,
            rows,
        )
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
    inserted_count = insert_employee_attrition_predictions()
    print(f"퇴사 예측 INSERT 완료: {inserted_count}건")


if __name__ == "__main__":
    main()
