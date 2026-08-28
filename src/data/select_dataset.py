import os

import pandas as pd
import pymysql
from dotenv import load_dotenv

from src.database.load_db import SELECT_DATASET_SQL


def select_dataframe(connection, parameters=None) -> pd.DataFrame:
    """MySQL 데이터베이스에서 데이터를 조회하여 데이터프레임으로 반환한다."""

    sql = SELECT_DATASET_SQL 

    with connection.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

    return pd.DataFrame(rows, columns=columns)

def main():
    load_dotenv()

    connection = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", "3306")),
        charset="utf8mb4",
    )

    try:
        train_dataframe = select_dataframe(
            connection,
            ("train",),
        )
        test_dataframe = select_dataframe(
            connection,
            ("test",),
        )

        print("train:", train_dataframe.shape)
        print("test:", test_dataframe.shape)

    finally:
        connection.close()


if __name__ == "__main__":
    main()