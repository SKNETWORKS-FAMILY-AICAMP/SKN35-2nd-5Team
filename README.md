# SKN35-2nd-5Team

## KT4 결제 고객 이탈 모델링 데이터셋

대용량 KT4 원본에서 결제 고객 23,789명의 모델링 데이터를 추출했습니다.
GitHub 단일 파일 제한을 넘지 않도록 원천 행동 로그는 결정론적 가중 샘플로
축소했고, 모델 학습용 집계 피처는 전체 로그에서 계산한 값을 보존했습니다.

### 최종 데이터

| 파일 | 내용 | 크기 |
|---|---|---:|
| `data/ednet_payment_users.csv` | 결제 고객 전원의 가중 행동 로그 샘플 1,007,413행 | 60.66 MB |
| `data/churn_modeling_features.csv` | 전체 로그 기반 14일 관측 피처 23,789행 × 35열 | 3.33 MB |
| `data/kt4_pass_expiry_repurchase_analysis.csv` | 이탈·재결제 라벨 요약 | 2.50 MB |
| `data/kt4_payment_transactions.csv` | 결제·환불·쿠폰 등록 트랜잭션 | 1.99 MB |

이탈 라벨 분포는 이탈 21,615명, 유지 2,174명입니다. `pay-only`는 유지 고객을
뜻하지 않습니다. 환불 없이 결제한 고객을 의미하며, pay-only 22,693명 중
20,519명은 비갱신 이탈 고객입니다.

### 행동 로그 샘플링

최종 행동 로그는 각 고객의 이벤트를 최초 결제 전, 결제 후 14일 관측 구간,
그 이후로 나눠 각각 최대 15개, 30개, 15개 일반 이벤트를 시간축에 균등하게
선택합니다. `pay`, `refund`, `enroll_coupon` 이벤트 27,781건은 전부 보존합니다.

`sample_weight`는 선택된 일반 이벤트가 같은 고객·기간 구간의 원본 이벤트를
몇 건 대표하는지 나타냅니다. 결제 관련 핵심 이벤트의 가중치는 1입니다.
이벤트 단위 통계를 학습에 사용할 때는 이 값을 표본 가중치로 사용할 수 있습니다.

실제 정답 키는 KT4 로그에 없으므로 정답률은 제공하지 않습니다.
`obs_response_with_answer_rate`는 답변 값이 기록된 응답 비율이며 정답률이 아닙니다.


## 프로젝트 구조 (Roadmap)

아래 구조는 데이터 분석부터 모델 학습, 설명, 리텐션 액션까지 확장하기 위한
최종 목표 구조입니다. 현재 저장소는 이 구조로 단계적으로 전환 중입니다.

```text
SKN35-2nd-5Team/
├─ app.py                                   # 🎯 메인 대시보드 엔트리포인트
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ data/                                    # 💾 정제된 핵심 데이터셋 (약 66 MB)
│  ├─ churn_modeling_features.csv           # ML/DL 모델링 전용 피처셋 (23,789 × 35)
│  ├─ ednet_payment_users.csv               # 결제 고객 행동 로그 통합본
│  └─ kt4_payment_transactions.csv          # 결제/환불 원천 트랜잭션
├─ artifacts/                               # 📦 모델/결과물/시각화 아티팩트
│  ├─ processed/                            # 처리된 중간 데이터 (.gitkeep)
│  ├─ models/
│  │  ├─ ml/                                # Decision Tree, Random Forest, XGBoost, LightGBM (.joblib)
│  │  └─ dl/                                # MLP (.joblib)
│  ├─ figures/                              # 차트 이미지 (.gitkeep)
│  └─ results/                              # ml_metrics.json, dl_metrics.json
├─ src/                                     # 🧩 핵심 소스코드 패키지
│  ├─ data/                                 # loader.py, validator.py
│  ├─ analysis/                             # eda.py
│  ├─ features/                             # churn.py, feature_engineering.py
│  ├─ clustering/                           # kmeans.py
│  ├─ ml/                                   # ML 모델, trainer.py, evaluate.py
│  ├─ dl/                                   # mlp.py, trainer.py, evaluate.py
│  ├─ comparison/                           # model_comparison.py
│  ├─ explain/                              # shap_analysis.py
│  ├─ retention/                            # risk_type.py, llm_client.py, retention_action.py
│  └─ utils/                                # constants.py, paths.py
├─ pages/                                   # 🖥️ Streamlit 9단계 멀티페이지 UI
│  ├─ 01_EDA.py                             # 01. 탐색적 데이터 분석
│  ├─ 02_Churn_Definition.py                # 02. 이탈 기준 및 타깃 정의
│  ├─ 03_Feature_Engineering.py             # 03. 피처 생성 및 상관관계
│  ├─ 04_User_Clustering.py                 # 04. 고객 행동 군집 분석
│  ├─ 05_ML_Training.py                     # 05. ML 4종 모델 학습 및 평가
│  ├─ 06_DL_Training.py                     # 06. 딥러닝(MLP) 모델 학습 및 평가
│  ├─ 07_Model_Comparison.py                # 07. 모델 종합 벤치마크 (ROC/PR Curve)
│  ├─ 08_SHAP_Analysis.py                   # 08. 모델 해석 및 중요 피처 분석
│  └─ 09_Retention_Action.py                # 09. 맞춤형 리텐션 액션 & LLM CRM 생성
├─ scripts/                                 # ⚡ CLI 일괄 실행 스크립트
│  ├─ run_eda.py
│  ├─ run_clustering.py
│  ├─ train_ml.py                           # ML 4종 일괄 학습
│  ├─ train_dl.py                           # DL 일괄 학습
│  └─ evaluate_models.py                    # 모델 랭킹 리더보드 출력
└─ tests/                                   # 🧪 단위 테스트 슈트 (목표: 100% Pass)
   ├─ fixtures.py
   ├─ test_data.py
   ├─ test_churn.py
   └─ test_features.py
```
