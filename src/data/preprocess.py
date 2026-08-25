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


# 타겟 컬럼(Attrition)을 명시적으로 매핑 : 이탈(Left)=1, 잔류(Stayed)=0
# LabelEncoder에 맡기면 알파벳순으로 Left=0, Stayed=1이 되어서
# f1_score가 기본으로 보는 pos_label=1이 "잔류"가 되어버리는 문제를 막기 위함
def encode_target(df: pd.DataFrame, target_col: str = "Attrition") -> pd.DataFrame:
    if target_col in df.columns and df[target_col].dtype in [
        "object",
        "string",
        "string[pyarrow]",
        "str",
    ]:
        df[target_col] = (df[target_col] == "Left").astype(int)
    return df


# 결측치를 문자형은 최빈값, 수치형은 중앙값으로 채움
# reference가 주어지면 (train 등) 그 통계값을 기준으로 채움 -> train/test 일관성 유지
def fill_missing_values(df: pd.DataFrame, reference: pd.DataFrame = None) -> pd.DataFrame:
    ref = reference if reference is not None else df
    for col in df.columns:
        if df[col].dtype in ["object", "string", "string[pyarrow]"]:
            df[col] = df[col].fillna(ref[col].mode()[0])
        else:
            df[col] = df[col].fillna(ref[col].median())
    return df


# 이진 범주형은 라벨 인코딩, 다중 범주형은 원핫 인코딩, bool은 int로 변환
# reference가 주어지면 (train 등) 어떤 컬럼을 binary/multi로 볼지, 어떤 매핑을 쓸지,
# 원핫 인코딩 후 최종 컬럼 구성이 어떻게 될지를 reference 기준으로 고정함
# Attrition(타겟)은 encode_target에서 이미 처리했으므로 여기서는 제외함
def encode_categorical(
    df: pd.DataFrame, reference: pd.DataFrame = None, target_col: str = "Attrition"
) -> pd.DataFrame:
    ref = reference if reference is not None else df

    le = LabelEncoder()
    binary_cols = [
        c
        for c in ref.columns
        if ref[c].dtype in ["object", "string", "string[pyarrow]"]
        and ref[c].nunique() == 2
        and c != target_col
    ]
    for col in binary_cols:
        le.fit(ref[col])
        df[col] = le.transform(df[col])

    multi_cols = [
        c
        for c in ref.columns
        if ref[c].dtype in ["object", "string", "string[pyarrow]"] and ref[c].nunique() > 2
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    if reference is not None:
        ref_dummy_cols = pd.get_dummies(ref, columns=multi_cols, drop_first=True).columns
        df = df.reindex(columns=ref_dummy_cols, fill_value=0)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


# 위 전처리 함수들을 순서대로 실행하는 전체 파이프라인
# reference가 주어지면 (train 등) reference에도 동일한 행 필터링/컬럼 삭제를 적용한 뒤
# 그 결과를 기준으로 결측치/인코딩 규칙을 df에 적용함
def preprocess_pipeline(df: pd.DataFrame, reference: pd.DataFrame = None) -> pd.DataFrame:
    df = remove_outliers(df)
    df = drop_unnecessary_columns(df)

    if reference is not None:
        reference = remove_outliers(reference)
        reference = drop_unnecessary_columns(reference)

    df = encode_target(df)
    if reference is not None:
        reference = encode_target(reference)

    df = fill_missing_values(df, reference=reference)
    df = encode_categorical(df, reference=reference)
    return df
