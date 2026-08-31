"""인사팀 화면(연봉 협상·팀 구성·인사 지원·안정도)에서 공통으로 쓰는 지표 계산.

원본 데이터에는 "인재 가치" 같은 점수가 없기 때문에, 이 모듈에서 납득 가능한 기준으로
직접 정의한다. 모든 점수는 0~100 스케일로 맞춰 서로 비교하거나 가중합할 수 있게 한다.
데이터의 값(영문 카테고리)은 이 모듈의 번역 딕셔너리를 거쳐 항상 한글로 화면에 표시한다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 한글 표시명 · 값 번역
# ---------------------------------------------------------------------------

TALENT_FEATURES = [
    "Education Level",
    "Performance Rating",
    "Company Reputation",
    "Company Tenure",
    "Leadership Opportunities",
]

TALENT_FEATURE_LABELS = {
    "Education Level": "학력",
    "Performance Rating": "성과 평가",
    "Company Reputation": "회사 평판",
    "Company Tenure": "총 경력연수",
    "Leadership Opportunities": "리더십 기회",
}

RISK_FEATURES = [
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Number of Promotions",
    "Gender",
    "Years at Company",
]

# 컬럼(피처) 이름 -> 한글 라벨
FEATURE_LABELS = {
    "Monthly Income": "월 소득",
    "Work-Life Balance": "일과 삶의 균형",
    "Job Satisfaction": "직무 만족도",
    "Number of Promotions": "승진 횟수",
    "Gender": "성별",
    "Years at Company": "현 직장 근속연수",
    "Overtime": "초과 근무",
    "Distance from Home": "출퇴근 거리",
    "Remote Work": "재택근무",
    "Job Role": "부서",
    "Job Level": "직급",
    "Company Size": "회사 규모",
    "Marital Status": "결혼 여부",
    "Number of Dependents": "부양가족 수",
    "Age": "나이",
    "Employee ID": "직원 ID",
    "Leadership Opportunities": "리더십 기회",
    "Innovation Opportunities": "혁신 기회",
    "Company Reputation": "회사 평판",
    "Employee Recognition": "직원 인정",
    "Attrition": "퇴사 여부",
    **TALENT_FEATURE_LABELS,
}

# 부서(Job Role) 값 번역 — 화면에는 항상 이 이름으로 표시한다.
DEPARTMENT_KR = {
    "Technology": "기술",
    "Finance": "금융",
    "Healthcare": "의료",
    "Media": "미디어",
    "Education": "교육",
}

# 직급(Job Level) 값 번역
LEVEL_KR = {
    "Entry": "신입",
    "Mid": "중간",
    "Senior": "시니어",
}
LEVEL_ORDER = ["Entry", "Mid", "Senior"]

# 그 외 범주형 값들의 공통 번역표 (여러 컬럼에서 재사용되는 값들)
VALUE_KR: dict[str, str] = {
    **DEPARTMENT_KR,
    **LEVEL_KR,
    "Male": "남성",
    "Female": "여성",
    "Married": "기혼",
    "Divorced": "이혼",
    "Single": "미혼",
    "Small": "소규모",
    "Medium": "중간",
    "Large": "대규모",
    "Yes": "예",
    "No": "아니요",
    "Low": "낮음",
    "Below Average": "평균 이하",
    "Average": "보통",
    "High": "높음",
    "Very High": "매우 높음",
    "Poor": "나쁨",
    "Fair": "보통",
    "Good": "좋음",
    "Excellent": "매우 좋음",
    "High School": "고졸",
    "Associate Degree": "전문학사",
    "Bachelor’s Degree": "학사",
    "Bachelor's Degree": "학사",
    "Master’s Degree": "석사",
    "Master's Degree": "석사",
    "PhD": "박사",
    "Left": "퇴사",
    "Stayed": "재직",
}


def translate(value: object) -> str:
    """영문 카테고리 값을 한글로 바꾼다. 매핑이 없으면 원래 값을 문자열로 반환한다."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    text = str(value)
    return VALUE_KR.get(text, text)


def translate_series(series: pd.Series) -> pd.Series:
    """시리즈 전체를 한글 값으로 바꾼다(존재하지 않는 값은 원문 유지)."""

    return series.astype(str).map(lambda v: VALUE_KR.get(v, v))


RISK_HIGH = 0.6
RISK_MID = 0.35

SATISFACTION_SCORE_MAP = {"Low": 25.0, "Medium": 50.0, "High": 75.0, "Very High": 100.0}
WORK_LIFE_SCORE_MAP = {"Poor": 25.0, "Fair": 50.0, "Good": 75.0, "Excellent": 100.0}
RECOGNITION_SCORE_MAP = {"Low": 25.0, "Medium": 50.0, "High": 75.0, "Very High": 100.0}

TALENT_VALUE_EXPLANATION = (
    "학력·성과 평가·회사 평판·총 경력연수·리더십 기회 5개 항목을 각각 0~100점으로 환산한 뒤 "
    "동일한 비중(20%씩)으로 평균한 값이에요. 특정 항목 하나에 점수가 쏠리지 않도록 다섯 관점을 "
    "고르게 반영했어요."
)


def _normalized_map(series: pd.Series, scores: dict[str, float], fallback: float = 50.0) -> pd.Series:
    return series.astype(str).map(scores).fillna(fallback).astype(float)


def add_talent_value(frame: pd.DataFrame) -> pd.DataFrame:
    """학력·성과·평판·경력·리더십 5개 항목을 동일 가중치로 환산해 0~100 인재 가치 지수를 만든다.

    가중치를 동일하게 둔 이유는 특정 항목(예: 성과 평가)에 점수가 쏠리지 않도록 하기
    위함이며, 다섯 항목 모두 "회사가 계속 투자할 가치가 있는 인재인가"를 각기 다른
    각도에서 보여주는 지표이기 때문이다.
    """

    result = frame.copy()
    result["학력 점수"] = _normalized_map(
        result["Education Level"],
        {
            "High School": 40,
            "Associate Degree": 55,
            "Bachelor’s Degree": 72,
            "Bachelor's Degree": 72,
            "Master’s Degree": 86,
            "Master's Degree": 86,
            "PhD": 100,
        },
    )
    result["성과 점수"] = _normalized_map(
        result["Performance Rating"],
        {"Low": 35, "Below Average": 45, "Average": 62, "High": 82, "Excellent": 100},
    )
    result["평판 점수"] = _normalized_map(
        result["Company Reputation"], {"Poor": 35, "Fair": 55, "Good": 78, "Excellent": 100}
    )
    tenure = pd.to_numeric(result["Company Tenure"], errors="coerce").fillna(0)
    result["경력 점수"] = (tenure.clip(lower=0, upper=30) / 30 * 100).round(1)
    result["리더십 점수"] = _normalized_map(result["Leadership Opportunities"], {"No": 45, "Yes": 100})

    score_columns = ["학력 점수", "성과 점수", "평판 점수", "경력 점수", "리더십 점수"]
    result["인재 가치 지수"] = result[score_columns].mean(axis=1).round(1)

    if "Job Satisfaction" in result.columns:
        result["만족도 점수"] = _normalized_map(result["Job Satisfaction"], SATISFACTION_SCORE_MAP)
    if "Work-Life Balance" in result.columns:
        result["워라밸 점수"] = _normalized_map(result["Work-Life Balance"], WORK_LIFE_SCORE_MAP)
    if "Employee Recognition" in result.columns:
        result["인정 점수"] = _normalized_map(result["Employee Recognition"], RECOGNITION_SCORE_MAP)

    return result


def risk_label(risk: float) -> str:
    if risk >= RISK_HIGH:
        return "높음"
    if risk >= RISK_MID:
        return "보통"
    return "낮음"


def risk_badge_tone(risk: float) -> str:
    """위험도에 따라 배지 색상 클래스를 반환한다."""

    if risk >= RISK_HIGH:
        return "danger"
    if risk >= RISK_MID:
        return "warning"
    return "safe"


# ---------------------------------------------------------------------------
# 직원 조회 보조 (부서 -> 직급 -> ID 순차 필터)
# ---------------------------------------------------------------------------


def department_options(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["Job Role"].dropna().unique().tolist())


def level_options(frame: pd.DataFrame, department: str | None = None) -> list[str]:
    subset = frame if department is None else frame[frame["Job Role"].eq(department)]
    present = set(subset["Job Level"].dropna().unique().tolist())
    return [level for level in LEVEL_ORDER if level in present]


def employee_label(row: pd.Series) -> str:
    return (
        f"ID {row['Employee ID']} · {translate(row['Job Role'])} · {translate(row['Job Level'])}"
    )


# ---------------------------------------------------------------------------
# 팀 구성
# ---------------------------------------------------------------------------


def team_fit_score(frame: pd.DataFrame, talent_weight: float = 0.6) -> pd.Series:
    """인재 가치와 잔류 가능성을 함께 반영한 팀 적합 점수(0~100)."""

    retention_weight = 1 - talent_weight
    return (
        frame["인재 가치 지수"] * talent_weight + (1 - frame["prediction"]) * 100 * retention_weight
    ).round(1)


def team_stability_verdict(mean_risk: float) -> tuple[str, str, str]:
    """평균 퇴사 위험을 바탕으로 팀 안정도 한 줄 판정을 반환한다.

    반환값: (라벨, 색상 톤, 설명 문구)
    """

    if mean_risk < 0.20:
        return "안정적", "safe", "팀 전체 퇴사 위험이 낮아 프로젝트를 안심하고 진행할 수 있어요."
    if mean_risk < 0.40:
        return "양호", "info", "위험 신호가 크지 않지만 핵심 인력의 만족도는 주기적으로 확인하세요."
    if mean_risk < 0.60:
        return "주의 필요", "warning", "위험도가 높은 팀원이 섞여 있어요. 대체 인력 후보를 함께 준비하세요."
    return "위험", "danger", "팀 평균 퇴사 위험이 높습니다. 구성 변경이나 리텐션 조치를 먼저 검토하세요."


def fill_team_slots(
    frame: pd.DataFrame,
    department: str,
    level_counts: dict[str, int],
    talent_weight: float = 0.6,
    exclude_ids: set | None = None,
) -> pd.DataFrame:
    """부서 + 직급별 필요 인원(level_counts)만큼, 팀 적합 점수가 높은 순으로 채운다."""

    exclude_ids = exclude_ids or set()
    picks = []
    for level, count in level_counts.items():
        if count <= 0:
            continue
        pool = frame[
            frame["Job Role"].eq(department)
            & frame["Job Level"].eq(level)
            & ~frame["Employee ID"].isin(exclude_ids)
        ].copy()
        if pool.empty:
            continue
        pool["팀 적합 점수"] = team_fit_score(pool, talent_weight)
        picks.append(pool.nlargest(count, "팀 적합 점수"))
    if not picks:
        return frame.iloc[0:0].copy()
    return pd.concat(picks, ignore_index=True)


def alternative_candidates(
    frame: pd.DataFrame,
    department: str,
    level: str,
    talent_weight: float,
    exclude_ids: set,
    limit: int = 8,
) -> pd.DataFrame:
    """같은 부서·직급 안에서 대체 후보를 적합 점수 순으로 보여준다(교체용)."""

    pool = frame[
        frame["Job Role"].eq(department)
        & frame["Job Level"].eq(level)
        & ~frame["Employee ID"].isin(exclude_ids)
    ].copy()
    if pool.empty:
        return pool
    pool["팀 적합 점수"] = team_fit_score(pool, talent_weight)
    return pool.nlargest(limit, "팀 적합 점수")


# ---------------------------------------------------------------------------
# 인사발령 · 승진 · 구조조정
# ---------------------------------------------------------------------------


def add_people_decision_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """동일 조건(같은 부서) 안에서 승진/구조조정/재배치 우선순위 점수를 매긴다.

    핵심 규칙: 인재 가치가 같다면 퇴사 예측률이 낮은 쪽에 승진 우선순위를 준다.
    """

    result = frame.copy()
    result["승진 우선 점수"] = (
        result["인재 가치 지수"] * 0.65 + (1 - result["prediction"]) * 100 * 0.35
    ).round(1)
    result["검토 우선 점수"] = (
        (100 - result["인재 가치 지수"]) * 0.55 + result["prediction"] * 100 * 0.45
    ).round(1)
    if "만족도 점수" in result.columns:
        # 인재 가치는 높은데 만족도가 낮으면 이직 신호일 수 있어 재배치 후보로 본다.
        result["재배치 신호 점수"] = (
            result["인재 가치 지수"] * 0.5
            + (100 - result["만족도 점수"]) * 0.3
            + result["prediction"] * 100 * 0.2
        ).round(1)
    return result


# ---------------------------------------------------------------------------
# 전사 안정도 · 피처별 퇴사율
# ---------------------------------------------------------------------------

DRIVER_FEATURE_CANDIDATES = [
    "Monthly Income",
    "Overtime",
    "Job Satisfaction",
    "Work-Life Balance",
    "Number of Promotions",
    "Years at Company",
    "Distance from Home",
    "Remote Work",
    "Job Level",
    "Company Size",
]


def attrition_rate_by_feature(frame: pd.DataFrame, feature: str, bins: int = 6) -> pd.DataFrame:
    """피처 값(또는 값 구간)별 실제 퇴사율과 인원수를 계산한다.

    수치형 피처는 표본 수가 비슷하도록 구간을 나누고(qcut), 범주형은 값을 한글로
    바꿔 그룹화한다.
    """

    if feature not in frame.columns:
        raise ValueError(f"존재하지 않는 컬럼입니다: {feature}")

    attrition = frame["Attrition"].astype(str).eq("Left")
    series = frame[feature]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric:
        numeric = pd.to_numeric(series, errors="coerce")
        try:
            grouped_key = pd.qcut(numeric, q=min(bins, numeric.nunique()), duplicates="drop")
        except ValueError:
            grouped_key = numeric
        labels = grouped_key.astype(str)
    else:
        labels = translate_series(series)

    table = (
        pd.DataFrame({"구간": labels, "퇴사여부": attrition})
        .groupby("구간", observed=True)["퇴사여부"]
        .agg(퇴사율="mean", 인원수="size")
        .reset_index()
    )

    if is_numeric:
        # 구간 문자열이 아니라 원래 순서로 정렬되도록 좌측 경계값을 기준 정렬한다.
        table["_정렬키"] = table["구간"].str.extract(r"\(([-\d\.]+),").astype(float)
        table = table.sort_values("_정렬키").drop(columns="_정렬키").reset_index(drop=True)
    elif feature == "Job Level":
        order = {LEVEL_KR[k]: i for i, k in enumerate(LEVEL_ORDER)}
        table = table.assign(_순서=table["구간"].map(order).fillna(99)).sort_values("_순서").drop(columns="_순서").reset_index(drop=True)
    else:
        table = table.sort_values("퇴사율", ascending=False).reset_index(drop=True)

    return table


def rank_attrition_drivers(
    frame: pd.DataFrame, candidates: list[str] | None = None
) -> pd.DataFrame:
    """후보 피처들을 구간별 퇴사율 편차(최대-최소) 기준으로 정렬해 상위 요인을 보여준다."""

    candidates = [c for c in (candidates or DRIVER_FEATURE_CANDIDATES) if c in frame.columns]
    rows = []
    for feature in candidates:
        try:
            table = attrition_rate_by_feature(frame, feature)
        except Exception:
            continue
        if table.empty:
            continue
        spread = float(table["퇴사율"].max() - table["퇴사율"].min())
        rows.append(
            {
                "피처": FEATURE_LABELS.get(feature, feature),
                "_원본": feature,
                "퇴사율 최대": float(table["퇴사율"].max()),
                "퇴사율 최소": float(table["퇴사율"].min()),
                "영향도(편차)": spread,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["피처", "_원본", "퇴사율 최대", "퇴사율 최소", "영향도(편차)"])
    return pd.DataFrame(rows).sort_values("영향도(편차)", ascending=False).reset_index(drop=True)


def department_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """부서별 인원·퇴사위험·인재가치·만족도를 한 번에 보여주는 안정도 표(부서명은 한글)."""

    agg = {
        "직원수": ("Employee ID", "size"),
        "평균_퇴사위험": ("prediction", "mean"),
        "평균_인재가치": ("인재 가치 지수", "mean"),
    }
    if "만족도 점수" in frame.columns:
        agg["평균_만족도"] = ("만족도 점수", "mean")

    table = (
        frame.groupby("Job Role", as_index=False)
        .agg(**agg)
        .sort_values("평균_퇴사위험", ascending=False)
        .rename(columns={"Job Role": "부서"})
    )
    table["부서"] = table["부서"].map(lambda v: DEPARTMENT_KR.get(v, v))
    return table


def stability_index(frame: pd.DataFrame) -> float:
    return float((1 - frame["prediction"].mean()) * 100)


# ---------------------------------------------------------------------------
# 깔끔한 슬라이더/시뮬레이션 범위 만들기
# ---------------------------------------------------------------------------


def _nice_step(raw_step: float) -> float:
    """1/2/5 × 10^n 중 raw_step 이상인 가장 작은 값을 고른다 (깔끔한 눈금 간격)."""

    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    for base in (1, 2, 5, 10):
        candidate = base * (10**exponent)
        if candidate >= raw_step:
            return float(candidate)
    return float(10 ** (exponent + 1))


def nice_range(min_value: float, max_value: float, target_steps: int = 24, is_integer: bool = True):
    """min~max를 넉넉하게 감싸는 깔끔한 구간과 step을 만든다.

    예: 실제 데이터가 1316~16149처럼 지저분해도, 1000~16500을 500 단위로 끊는 식으로
    보기 좋은 범위를 돌려준다. 반환값은 (정리된 최소값, 정리된 최대값, step).
    """

    span = max(max_value - min_value, 1.0)
    raw_step = span / max(target_steps, 1)
    step = _nice_step(raw_step)
    if is_integer:
        step = max(1, round(step))
        lo = int(math.floor(min_value / step) * step)
        hi = int(math.ceil(max_value / step) * step)
    else:
        lo = math.floor(min_value / step) * step
        hi = math.ceil(max_value / step) * step
    return lo, hi, step


def nice_wheel_options(min_value: float, max_value: float, current, target_steps: int = 24):
    """nice_range로 만든 구간의 옵션 목록을 만들고, 현재 값이 목록에 없으면 끼워 넣는다."""

    is_integer = isinstance(current, (int, np.integer))
    lo, hi, step = nice_range(float(min_value), float(max_value), target_steps, is_integer)
    if is_integer:
        values = list(range(int(lo), int(hi) + int(step), int(step)))
        values = [int(v) for v in values]
    else:
        count = int(round((hi - lo) / step)) + 1
        values = [round(lo + i * step, 2) for i in range(count)]
    if current not in values:
        values.append(current)
    return sorted(set(values))


__all__ = [
    "TALENT_FEATURES",
    "TALENT_FEATURE_LABELS",
    "RISK_FEATURES",
    "FEATURE_LABELS",
    "DEPARTMENT_KR",
    "LEVEL_KR",
    "LEVEL_ORDER",
    "VALUE_KR",
    "translate",
    "translate_series",
    "RISK_HIGH",
    "RISK_MID",
    "SATISFACTION_SCORE_MAP",
    "TALENT_VALUE_EXPLANATION",
    "DRIVER_FEATURE_CANDIDATES",
    "add_talent_value",
    "risk_label",
    "risk_badge_tone",
    "department_options",
    "level_options",
    "employee_label",
    "team_fit_score",
    "team_stability_verdict",
    "fill_team_slots",
    "alternative_candidates",
    "add_people_decision_scores",
    "attrition_rate_by_feature",
    "rank_attrition_drivers",
    "department_overview",
    "stability_index",
    "nice_range",
    "nice_wheel_options",
]
