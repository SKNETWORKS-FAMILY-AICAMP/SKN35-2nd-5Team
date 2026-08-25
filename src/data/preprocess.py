# 데이터 전처리 함수 정의

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 재직 연수(YearsAtCompany)가 0~60 범위를 벗어나는 이상치 행 제거
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "YearsAtCompany" in df.columns:
        df = df[(df["YearsAtCompany"] >= 0) & (df["YearsAtCompany"] <= 60)]
    elif "Years at Company" in df.columns:
        df = df[(df["Years at Company"] >= 0) & (df["Years at Company"] <= 60)]
    return df


# ID나 상수값처럼 예측에 도움 안 되는 컬럼 삭제
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["Employee ID", "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# 결측치를 문자형은 최빈값, 수치형은 중앙값으로 채움
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ["object", "string", "string[pyarrow]"]:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


# 이진 범주형은 라벨 인코딩, 다중 범주형은 원핫 인코딩, bool은 int로 변환
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    binary_cols = [
        c
        for c in df.columns
        if df[c].dtype in ["object", "string", "string[pyarrow]"] and df[c].nunique() == 2
    ]
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    multi_cols = [
        c
        for c in df.columns
        if df[c].dtype in ["object", "string", "string[pyarrow]"] and df[c].nunique() > 2
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


# 위 전처리 함수들을 순서대로 실행하는 전체 파이프라인
def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_outliers(df)
    df = drop_unnecessary_columns(df)
    df = fill_missing_values(df)
    df = encode_categorical(df)
    return df
