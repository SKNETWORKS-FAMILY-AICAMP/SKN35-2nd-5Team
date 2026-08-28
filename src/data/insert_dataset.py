from pathlib import Path
import os

import pandas as pd
import pymysql
from dotenv import load_dotenv

from src.database.send_db import INSERT_DATASET_SQL
from src.data.loader import load_raw_test, load_raw_train
from src.data.preprocess import preprocess_pipeline

SQL_PATH = Path("src/database/send_db.py")  # SQL 쿼리 파일 경로


def insert_dataframe(connection, dataframe: pd.DataFrame) -> None:
    """데이터프레임을 MySQL 데이터베이스에 삽입한다."""

    sql = INSERT_DATASET_SQL  # SQL 쿼리 가져오기

    rows = [
        tuple(None if pd.isna(value) else value for value in row)
        for row in dataframe.itertuples(index=False, name=None)
    ]

    with connection.cursor() as cursor:
        cursor.executemany(sql, rows)



def main():
    load_dotenv()  # .env 파일에서 환경 변수 로드

    train_raw = load_raw_train()
    test_raw = load_raw_test()

    train_processed = preprocess_pipeline(train_raw)
    test_processed = preprocess_pipeline(test_raw, reference=train_raw)

    # MySQL 연결 설정
    connection = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
    )

    try:
        insert_dataframe(connection, train_processed)
        insert_dataframe(connection, test_processed)
        connection.commit()

    except Exception as e:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
