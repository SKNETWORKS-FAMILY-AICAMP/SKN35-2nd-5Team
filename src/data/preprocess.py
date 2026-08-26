import pandas as pd

# ==============================================================================
# 1. Ordinal Encoding Mapping
# ==============================================================================

ORDINAL_MAPS = {
    "Job Level": {
        "Entry": 0,
        "Mid": 1,
        "Senior": 2,
    },
    "Company Size": {
        "Small": 0,
        "Medium": 1,
        "Large": 2,
    },
    "Company Reputation": {
        "Poor": 0,
        "Fair": 1,
        "Good": 2,
        "Excellent": 3,
    },
    "Work-Life Balance": {
        "Poor": 0,
        "Fair": 1,
        "Good": 2,
        "Excellent": 3,
    },
    "Employee Recognition": {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Very High": 3,
    },
    "Job Satisfaction": {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Very High": 3,
    },
    "Performance Rating": {
        "Low": 0,
        "Below Average": 1,
        "Average": 2,
        "High": 3,
    },
    "Education Level": {
        "High School": 0,
        "Associate Degree": 1,
        "Bachelor’s Degree": 2,
        "Master’s Degree": 3,
        "PhD": 4,
    },
}


# ==============================================================================
# 2. Binary Encoding Mapping
# ==============================================================================

BINARY_MAPS = {
    "Gender": {
        "Female": 0,
        "Male": 1,
    },
    "Overtime": {
        "No": 0,
        "Yes": 1,
    },
    "Remote Work": {
        "No": 0,
        "Yes": 1,
    },
    "Leadership Opportunities": {
        "No": 0,
        "Yes": 1,
    },
    "Innovation Opportunities": {
        "No": 0,
        "Yes": 1,
    },
}


# ==============================================================================
# 3. 문자열 정규화
# ==============================================================================


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    문자열 컬럼의 공백과 따옴표 표현을 정규화합니다.

    예:
        " Male " -> "Male"
        "Bachelor's Degree" -> "Bachelor’s Degree"
    """

    df = df.copy()

    string_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in string_columns:
        df[col] = df[col].astype("string").str.strip().str.replace("'", "’", regex=False)

    return df


# ==============================================================================
# 4. 이상치 제거
# ==============================================================================


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Years at Company의 명백한 범위 밖 데이터를 제거합니다.

    지원 컬럼명:
        Years at Company
        YearsAtCompany
    """

    df = df.copy()

    if "YearsAtCompany" in df.columns:
        df = df[(df["YearsAtCompany"] >= 0) & (df["YearsAtCompany"] <= 60)]

    elif "Years at Company" in df.columns:
        df = df[(df["Years at Company"] >= 0) & (df["Years at Company"] <= 60)]

    return df


# ==============================================================================
# 5. 불필요한 컬럼 제거
# ==============================================================================


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습에 필요하지 않은 식별자/상수 컬럼을 제거합니다.
    """

    df = df.copy()

    cols_to_drop = [
        "Employee ID",
        "EmployeeNumber",
        "EmployeeCount",
        "StandardHours",
        "Over18",
    ]

    existing_cols = [col for col in cols_to_drop if col in df.columns]

    return df.drop(columns=existing_cols)


# ==============================================================================
# 6. Target Encoding
# ==============================================================================


def encode_target(
    df: pd.DataFrame,
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """
    Attrition:
        Stayed -> 0
        Left   -> 1

    이미 숫자형이면 그대로 유지합니다.
    """

    df = df.copy()

    if target_col not in df.columns:
        return df

    if pd.api.types.is_numeric_dtype(df[target_col]):
        return df

    normalized = df[target_col].astype("string").str.strip()

    valid_values = set(normalized.dropna().unique())

    allowed_values = {
        "Stayed",
        "Left",
    }

    unknown_values = sorted(valid_values - allowed_values)

    if unknown_values:
        raise ValueError(f"'{target_col}' 컬럼에 예상하지 못한 값이 있습니다: {unknown_values}")

    df[target_col] = (normalized == "Left").astype(int)

    return df


# ==============================================================================
# 7. 결측치 처리
# ==============================================================================


def fill_missing_values(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    결측치를 처리합니다.

    reference가 있으면 reference의 통계량을 사용합니다.

    문자열:
        최빈값

    숫자:
        중앙값
    """

    df = df.copy()

    ref = reference if reference is not None else df

    for col in df.columns:
        if col not in ref.columns:
            continue

        # ----------------------------------------------------------------------
        # 문자열 / 범주형
        # ----------------------------------------------------------------------

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            mode = ref[col].mode(dropna=True)

            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

        # ----------------------------------------------------------------------
        # 숫자형
        # ----------------------------------------------------------------------

        else:
            median = ref[col].median()

            if pd.notna(median):
                df[col] = df[col].fillna(median)

    return df


# ==============================================================================
# 8. Feature Engineering
# ==============================================================================


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HR 도메인 기반 파생 변수를 생성합니다.

    생성 feature:

    1. Industry Experience Gap
       = Company Tenure - Years at Company

    2. Promotion Rate
       = Number of Promotions / (Years at Company + 1)
    """

    df = df.copy()

    # --------------------------------------------------------------------------
    # Years at Company 컬럼명 확인
    # --------------------------------------------------------------------------

    years_at_company_col = None

    if "Years at Company" in df.columns:
        years_at_company_col = "Years at Company"

    elif "YearsAtCompany" in df.columns:
        years_at_company_col = "YearsAtCompany"

    # --------------------------------------------------------------------------
    # Company Tenure 컬럼 확인
    # --------------------------------------------------------------------------

    company_tenure_col = None

    if "Company Tenure" in df.columns:
        company_tenure_col = "Company Tenure"

    # --------------------------------------------------------------------------
    # 1. Industry Experience Gap
    # --------------------------------------------------------------------------

    if years_at_company_col is not None and company_tenure_col is not None:
        df["Industry Experience Gap"] = df[company_tenure_col] - df[years_at_company_col]

        # 음수는 논리적으로 허용하지 않음
        df["Industry Experience Gap"] = df["Industry Experience Gap"].clip(lower=0)

    # --------------------------------------------------------------------------
    # 2. Promotion Rate
    # --------------------------------------------------------------------------

    if years_at_company_col is not None and "Number of Promotions" in df.columns:
        df["Promotion Rate"] = df["Number of Promotions"] / (df[years_at_company_col] + 1)

    return df


# ==============================================================================
# 9. Ordinal Encoding
# ==============================================================================


def encode_ordinal(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    순서가 존재하는 범주형 데이터를 숫자로 변환합니다.
    """

    df = df.copy()

    for col, mapping in ORDINAL_MAPS.items():
        if col not in df.columns:
            continue

        normalized = df[col].astype("string").str.strip().str.replace("'", "’", regex=False)

        mapped = normalized.map(mapping)

        # ----------------------------------------------------------------------
        # mapping되지 않은 값 검증
        # ----------------------------------------------------------------------

        unmapped_mask = mapped.isna() & df[col].notna()

        if unmapped_mask.any():
            unmapped_values = sorted(
                df.loc[
                    unmapped_mask,
                    col,
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(f"'{col}' 컬럼에 ORDINAL_MAPS에 없는 값이 있습니다: {unmapped_values}")

        df[col] = mapped

    return df


# ==============================================================================
# 10. Binary Encoding
# ==============================================================================


def encode_binary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Binary categorical 데이터를 0/1로 변환합니다.
    """

    df = df.copy()

    for col, mapping in BINARY_MAPS.items():
        if col not in df.columns:
            continue

        normalized = df[col].astype("string").str.strip()

        mapped = normalized.map(mapping)

        # ----------------------------------------------------------------------
        # mapping되지 않은 값 검증
        # ----------------------------------------------------------------------

        unmapped_mask = mapped.isna() & df[col].notna()

        if unmapped_mask.any():
            unmapped_values = sorted(
                df.loc[
                    unmapped_mask,
                    col,
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(f"'{col}' 컬럼에 BINARY_MAPS에 없는 값이 있습니다: {unmapped_values}")

        df[col] = mapped

    return df


# ==============================================================================
# 11. One-Hot Encoding
# ==============================================================================


def encode_categorical(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """
    순서가 없는 다중 범주형 데이터를 One-Hot Encoding합니다.

    reference가 있으면 reference의 column 구조를 기준으로
    현재 데이터의 column을 정렬합니다.
    """

    df = df.copy()

    ref = reference if reference is not None else df

    # --------------------------------------------------------------------------
    # 문자열 categorical 컬럼 탐색
    # --------------------------------------------------------------------------

    categorical_cols = ref.select_dtypes(include=["object", "string"]).columns.tolist()

    # --------------------------------------------------------------------------
    # One-Hot 대상
    #
    # - target 제외
    # - ordinal 제외
    # - binary 제외
    # - 3개 이상의 category
    # --------------------------------------------------------------------------

    multi_cols = [
        col
        for col in categorical_cols
        if col != target_col
        and col not in ORDINAL_MAPS
        and col not in BINARY_MAPS
        and ref[col].nunique(dropna=True) > 2
    ]

    # --------------------------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------------------------

    if multi_cols:
        df = pd.get_dummies(
            df,
            columns=multi_cols,
            drop_first=True,
            dtype=int,
        )

    # --------------------------------------------------------------------------
    # Train 기준 column 구조 유지
    # --------------------------------------------------------------------------

    if reference is not None:
        ref_encoded = pd.get_dummies(
            ref,
            columns=multi_cols,
            drop_first=True,
            dtype=int,
        )

        reference_columns = ref_encoded.columns

        df = df.reindex(
            columns=reference_columns,
            fill_value=0,
        )

    # --------------------------------------------------------------------------
    # Boolean → int
    # --------------------------------------------------------------------------

    bool_cols = df.select_dtypes(include=["bool"]).columns

    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    return df


# ==============================================================================
# 12. 최종 데이터 타입 정리
# ==============================================================================


def finalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    최종적으로 모델에 넣을 수 있는 형태로 데이터 타입을 정리합니다.
    """

    df = df.copy()

    # Boolean → int
    bool_cols = df.select_dtypes(include=["bool"]).columns

    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # 숫자형 데이터 정리
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


# ==============================================================================
# 13. 전체 전처리 Pipeline
# ==============================================================================


def preprocess_pipeline(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    원본 DataFrame을 전처리하여 반환합니다.

    처리 순서:

        1. 문자열 정규화
        2. 이상치 제거
        3. 불필요한 컬럼 제거
        4. Target Encoding
        5. 결측치 처리
        6. Feature Engineering
        7. Ordinal Encoding
        8. Binary Encoding
        9. One-Hot Encoding
        10. 데이터 타입 정리

    주의:

        StandardScaler
        PolynomialFeatures
        PCA

        는 여기서 처리하지 않습니다.

        이들은 모델 학습 과정에서 Train 데이터에만
        fit하도록 분리합니다.
    """

    # --------------------------------------------------------------------------
    # 원본 보호
    # --------------------------------------------------------------------------

    df = df.copy()

    if reference is not None:
        reference = reference.copy()

    # --------------------------------------------------------------------------
    # 1. 문자열 정규화
    # --------------------------------------------------------------------------

    df = normalize_string_columns(df)

    if reference is not None:
        reference = normalize_string_columns(reference)

    # --------------------------------------------------------------------------
    # 2. 이상치 제거
    # --------------------------------------------------------------------------

    df = remove_outliers(df)

    if reference is not None:
        reference = remove_outliers(reference)

    # --------------------------------------------------------------------------
    # 3. 불필요한 컬럼 제거
    # --------------------------------------------------------------------------

    df = drop_unnecessary_columns(df)

    if reference is not None:
        reference = drop_unnecessary_columns(reference)

    # --------------------------------------------------------------------------
    # 4. Target Encoding
    # --------------------------------------------------------------------------

    df = encode_target(df)

    if reference is not None:
        reference = encode_target(reference)

    # --------------------------------------------------------------------------
    # 5. 결측치 처리
    # --------------------------------------------------------------------------

    df = fill_missing_values(
        df,
        reference=reference,
    )

    if reference is not None:
        reference = fill_missing_values(reference)

    # --------------------------------------------------------------------------
    # 6. Feature Engineering
    # --------------------------------------------------------------------------

    df = create_features(df)

    if reference is not None:
        reference = create_features(reference)

    # --------------------------------------------------------------------------
    # 7. Ordinal Encoding
    # --------------------------------------------------------------------------

    df = encode_ordinal(df)

    if reference is not None:
        reference = encode_ordinal(reference)

    # --------------------------------------------------------------------------
    # 8. Binary Encoding
    # --------------------------------------------------------------------------

    df = encode_binary(df)

    if reference is not None:
        reference = encode_binary(reference)

    # --------------------------------------------------------------------------
    # 9. One-Hot Encoding
    # --------------------------------------------------------------------------

    df = encode_categorical(
        df,
        reference=reference,
    )

    # --------------------------------------------------------------------------
    # 10. 최종 데이터 타입 정리
    # --------------------------------------------------------------------------

    df = finalize_dataframe(df)

    return df
