import pandas as pd
from dotenv import load_dotenv

from src.data.loader import load_raw_test, load_raw_train
from src.data.preprocess import preprocess_pipeline

from .db import get_db_connection

load_dotenv()


COLUMN_MAPPING = {
    "Employee ID": "employee_id",
    "Age": "age",
    "Gender": "gender",
    "Years at Company": "years_at_company",
    "Job Role": "job_role",
    "Monthly Income": "monthly_income",
    "Work-Life Balance": "work_life_balance",
    "Job Satisfaction": "job_satisfaction",
    "Performance Rating": "performance_rating",
    "Number of Promotions": "number_of_promotions",
    "Overtime": "overtime",
    "Distance from Home": "distance_from_home",
    "Education Level": "education_level",
    "Marital Status": "marital_status",
    "Number of Dependents": "number_of_dependents",
    "Job Level": "job_level",
    "Company Size": "company_size",
    "Company Tenure": "company_tenure",
    "Remote Work": "remote_work",
    "Leadership Opportunities": "leadership_opportunities",
    "Innovation Opportunities": "innovation_opportunities",
    "Company Reputation": "company_reputation",
    "Employee Recognition": "employee_recognition",
    "Attrition": "attrition",
    "Industry Experience Gap": "industry_experience_gap",
    "Promotion Rate": "promotion_rate",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV 컬럼명을 DB 컬럼명으로 변경합니다.
    """
    return df.rename(columns=COLUMN_MAPPING)


def insert_employee_attrition_raw(
    data_type: str,
    table_name: str = "employee_attrition_raw",
):
    """
    Raw CSV 데이터를 train/test 유형과 함께 MySQL에 저장합니다.
    """

    if data_type not in {"train", "test"}:
        raise ValueError("data_type은 'train' 또는 'test'만 가능합니다.")

    df = load_raw_train() if data_type == "train" else load_raw_test()

    print(f"CSV 데이터 : {len(df)}건")
    print(f"데이터 유형 : {data_type}")

    df = rename_columns(df)

    columns = [
        "employee_id",
        "age",
        "gender",
        "years_at_company",
        "job_role",
        "monthly_income",
        "work_life_balance",
        "job_satisfaction",
        "performance_rating",
        "number_of_promotions",
        "overtime",
        "distance_from_home",
        "education_level",
        "marital_status",
        "number_of_dependents",
        "job_level",
        "company_size",
        "company_tenure",
        "remote_work",
        "leadership_opportunities",
        "innovation_opportunities",
        "company_reputation",
        "employee_recognition",
        "attrition",
    ]

    df = df[columns]

    duplicate_count = df["employee_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Employee ID 중복 발견 : {duplicate_count}건")

    df = df.astype(object).where(
        pd.notna(df),
        None,
    )

    data = [
        (
            row.employee_id,
            data_type,
            row.age,
            row.gender,
            row.years_at_company,
            row.job_role,
            row.monthly_income,
            row.work_life_balance,
            row.job_satisfaction,
            row.performance_rating,
            row.number_of_promotions,
            row.overtime,
            row.distance_from_home,
            row.education_level,
            row.marital_status,
            row.number_of_dependents,
            row.job_level,
            row.company_size,
            row.company_tenure,
            row.remote_work,
            row.leadership_opportunities,
            row.innovation_opportunities,
            row.company_reputation,
            row.employee_recognition,
            row.attrition,
        )
        for row in df.itertuples(
            index=False,
            name="Employee",
        )
    ]

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        query = f"""
            INSERT INTO `{table_name}` (
                employee_id,
                type,
                age,
                gender,
                years_at_company,
                job_role,
                monthly_income,
                work_life_balance,
                job_satisfaction,
                performance_rating,
                number_of_promotions,
                overtime,
                distance_from_home,
                education_level,
                marital_status,
                number_of_dependents,
                job_level,
                company_size,
                company_tenure,
                remote_work,
                leadership_opportunities,
                innovation_opportunities,
                company_reputation,
                employee_recognition,
                attrition
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
        """

        print(f"INSERT 시작 : {table_name} ({data_type}, {len(data)}건)")

        cursor.executemany(query, data)

        connection.commit()

        print(f"INSERT 완료 : {cursor.rowcount}건")

    except Exception:
        if connection:
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()


def insert_employee_attrition_processed(
    data_type: str,
    table_name: str = "employee_attrition_processed",
):
    """
    Raw CSV를 preprocess_pipeline으로 전처리한 후
    train/test 유형과 함께 MySQL에 저장합니다.
    """

    if data_type not in {"train", "test"}:
        raise ValueError("data_type은 'train' 또는 'test'만 가능합니다.")

    raw_df = load_raw_train() if data_type == "train" else load_raw_test()

    print(f"원본 데이터 : {len(raw_df)}건")
    print(f"데이터 유형 : {data_type}")

    if "Employee ID" not in raw_df.columns:
        raise ValueError("CSV에 'Employee ID' 컬럼이 없습니다.")

    employee_ids = raw_df["Employee ID"].copy()

    if data_type == "train":
        processed_df = preprocess_pipeline(raw_df)

    else:
        train_raw = load_raw_train()

        processed_df = preprocess_pipeline(
            raw_df,
            reference=train_raw,
        )

    processed_df.insert(
        0,
        "employee_id",
        employee_ids.loc[processed_df.index].values,
    )

    processed_df.insert(
        1,
        "type",
        data_type,
    )

    processed_df = processed_df.reset_index(drop=True)

    processed_df = rename_columns(processed_df)

    print(f"전처리 데이터 : {len(processed_df)}건")
    print(f"DB 저장 컬럼 수 : {len(processed_df.columns)}개")

    print("\n[Processed Columns]")

    for index, column in enumerate(
        processed_df.columns,
        start=1,
    ):
        print(f"{index:02d}. {column}")

    duplicate_count = processed_df["employee_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Employee ID 중복 발견 : {duplicate_count}건")

    if processed_df.columns.duplicated().any():
        duplicated_columns = processed_df.columns[processed_df.columns.duplicated()].tolist()

        raise ValueError(f"중복된 컬럼명이 존재합니다: {duplicated_columns}")

    processed_df = processed_df.astype(object).where(
        pd.notna(processed_df),
        None,
    )

    processed_df.columns = (
        processed_df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    columns = list(processed_df.columns)

    data = [
        tuple(row)
        for row in processed_df.itertuples(
            index=False,
            name=None,
        )
    ]

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # 컬럼명에 백틱을 적용합니다.
        #
        # 예:
        # `employee_id`, `type`, `years_at_company`
        #
        # 이렇게 하면 MySQL 예약어와 충돌하는 것을 방지할 수 있습니다.
        column_sql = ",\n                ".join(f"`{column}`" for column in columns)

        placeholder_sql = ", ".join(["%s"] * len(columns))

        query = f"""
            INSERT INTO `{table_name}` (
                {column_sql}
            )
            VALUES (
                {placeholder_sql}
            )
        """

        print(f"\nINSERT 시작 : {table_name} ({data_type}, {len(data)}건)")

        cursor.executemany(
            query,
            data,
        )

        connection.commit()

        print(f"INSERT 완료 : {cursor.rowcount}건")

    except Exception:
        if connection:
            connection.rollback()

        raise

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()


if __name__ == "__main__":
    insert_employee_attrition_raw(
        data_type="train",
    )

    insert_employee_attrition_raw(
        data_type="test",
    )

    insert_employee_attrition_processed(
        data_type="train",
    )

    insert_employee_attrition_processed(
        data_type="test",
    )
