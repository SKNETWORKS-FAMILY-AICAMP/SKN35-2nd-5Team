"""Exploratory data analysis page."""

import matplotlib.pyplot as plt
import streamlit as st

from src.analysis.eda import (
    attrition_rate_by,
    build_eda_report,
    categorical_summary,
    numeric_summary,
)
from src.load_data.loader import load_train_data
from src.utils.constants import ID_COLUMN, TARGET_COLUMN

st.set_page_config(page_title="01 EDA", page_icon="🔎", layout="wide")
st.title("01. 탐색적 데이터 분석")


@st.cache_data
def get_data():
    return load_train_data()


data = get_data()
report = build_eda_report(data)
summary_columns = st.columns(5)
summary_columns[0].metric("행", f"{report['rows']:,}")
summary_columns[1].metric("열", report["columns"])
summary_columns[2].metric("이탈률", f"{report['attrition_rate']:.1%}")
summary_columns[3].metric("결측치", report["missing_values"])
summary_columns[4].metric("중복 행", report["duplicate_rows"])

left, right = st.columns(2)
with left:
    st.subheader("타깃 분포")
    st.bar_chart(data[TARGET_COLUMN].value_counts())
with right:
    st.subheader("데이터 미리보기")
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)

numeric_columns = data.select_dtypes(include="number").columns.drop(ID_COLUMN).tolist()
selected_numeric = st.selectbox("수치형 분포", numeric_columns)
figure, axis = plt.subplots(figsize=(10, 3.5))
for label, group in data.groupby(TARGET_COLUMN):
    axis.hist(group[selected_numeric], bins=25, alpha=0.55, label=label)
axis.set_xlabel(selected_numeric)
axis.set_ylabel("Count")
axis.legend()
st.pyplot(figure)
plt.close(figure)

categorical_columns = [
    column for column in data.select_dtypes(exclude="number").columns if column != TARGET_COLUMN
]
selected_category = st.selectbox("그룹별 이탈률", categorical_columns)
rates = attrition_rate_by(data, selected_category)
st.bar_chart(rates.set_index(selected_category)["attrition_rate"])
st.dataframe(rates, use_container_width=True, hide_index=True)

with st.expander("상세 기술통계"):
    st.markdown("**수치형**")
    st.dataframe(numeric_summary(data), use_container_width=True, hide_index=True)
    st.markdown("**범주형**")
    st.dataframe(categorical_summary(data), use_container_width=True, hide_index=True)
