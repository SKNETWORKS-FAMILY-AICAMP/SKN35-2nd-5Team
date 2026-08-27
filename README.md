# Employee Attrition Modeling Lab

직원 이탈 데이터를 탐색하고, ML 4종을 동일한 기준으로 학습·비교한 뒤
모델 해석과 리텐션 검토 후보 산출까지 연결하는 Streamlit 프로젝트입니다.

현재 단계는 **튜닝 전 기준선 프레임워크**입니다. Hugging Face나 사전학습 모델을
사용하지 않으며, 모든 모델은 \`data/train.csv\`에서 처음부터 학습합니다.

## 데이터

- 학습 파일: \`data/train.csv\`
- 검증/스코어링 파일: \`data/test.csv\`
- 타깃: \`Attrition\` (\`Left=1\`, \`Stayed=0\`)
- 식별자: \`Employee ID\` — 모델 입력에서 자동 제외

데이터 로더는 파일 존재 여부, 필수 컬럼, 직원 ID 중복, 타깃 레이블을 검증합니다.

## 구성

# 프로젝트 구조

```text
.
├── README.md                         # 프로젝트 설명 및 실행 방법
│
├── artifacts/                        # 학습 결과물 저장
│   ├── dl/                           # 딥러닝 모델 결과물
│   │   ├── mlp_best_params.pkl       # MLP 최적 하이퍼파라미터
│   │   ├── mlp_model.pt              # 학습된 MLP 모델
│   │   ├── mlp_scaler.pkl            # MLP 전처리에 사용한 Scaler
│   │   ├── mlp_threshold.pkl         # MLP 최적 분류 임계값
│   │   └── mlp_metadata.pkl          # MLP 입력 피처 및 설정 메타데이터
│   │
│   ├── ml/                           # 머신러닝 모델 결과물
│   │   ├── best_ml_model.joblib      # 최종 선정 ML 모델
│   │   └── {model}.joblib            # 모델별 학습 결과
│   │
│   └── reports/                      # ML/DL 공통 성능 리포트
│       ├── ml_leaderboard.csv        # ML 모델별 검증 성능
│       ├── best_ml_test_metrics.csv  # 최고 ML 모델 테스트 성능
│       └── dl_metrics.csv            # DL 모델 테스트 성능
│
├── data/                             # 프로젝트 데이터
│   ├── preprocessing/                # 전처리된 데이터
│   │   └── train_processed.csv       # 전처리 완료된 학습 데이터
│   │
│   └── raw/                          # 원본 데이터
│       ├── test.csv                  # 테스트용 원본 데이터
│       └── train.csv                 # 학습용 원본 데이터
│
├── main.py                           # Streamlit 애플리케이션 실행 진입점
│
├── notebooks/                        # 데이터 분석 및 모델 실험용 Jupyter Notebook
│   ├── dl/                           # 딥러닝 실험
│   │   └── mlp.ipynb                 # MLP 모델 실험 및 학습 과정
│   │
│   ├── ml/                           # 머신러닝 실험
│   │
│   └── preprocessing/                # 데이터 전처리 실험
│       └── processing.ipynb          # 데이터 탐색 및 전처리 과정
│
├── pyproject.toml                    # Python 프로젝트 설정 및 의존성 관리
│
├── src/                              # 실제 프로젝트에서 사용하는 소스 코드
│   ├── __init__.py                   # Python 패키지 초기화
│   ├── config.py                     # 프로젝트 전반의 설정값 관리
│   │
│   ├── data/                         # 데이터 처리 관련 코드
│   │   ├── loader.py                 # 데이터 로딩
│   │   └── preprocess.py             # 데이터 전처리 로직
│   │
│   ├── models/                       # 머신러닝 / 딥러닝 모델 관련 코드
│   │   ├── dl/                       # 딥러닝
│   │   │   ├── mlp_model.py          # MLP 모델 구조 정의
│   │   │   └── train.py              # 딥러닝 모델 학습 흐름
│   │   │   └── predict.py            # 딥러닝 모델 예측
│   │   │
│   │   └── ml/                       # 머신러닝 공통 학습 및 모델별 구현
│   │       ├── __init__.py           # ML 패키지 초기화
│   │       ├── train.py              # ML 학습, 비교, 최고 모델 재학습 흐름
│   │       ├── logistic_regression.py # Logistic Regression 모델 정의
│   │       ├── random_forest.py      # Random Forest 모델 정의
│   │       ├── xgboost.py            # XGBoost 모델 정의
│   │       ├── xgboost_tuning.py     # XGBoost GPU 하이퍼파라미터 튜닝
│   │       └── lightgbm.py           # LightGBM 모델 정의
│   │
│   ├── pages/                        # Streamlit 페이지
│   │   ├── 01_EDA.py                 # 데이터 탐색 및 EDA 화면
│   │   ├── 02_ML_Training.py         # 머신러닝 학습 화면
│   │   ├── 03_Model_Comparison.py    # 모델 성능 비교 화면
│   │   ├── 08_SHAP_Analysis.py       # SHAP 기반 모델 해석 화면
│   │   └── 09_Retention_Action.py    # 이탈 예측 결과 및 대응 전략 화면
│   │
│   └── utils/                        # 여러 곳에서 공통으로 사용하는 기능
│       ├── __init__.py               # Python 패키지 초기화
│       ├── constants.py              # 공통 상수 및 설정값
│       ├── metrics.py                # ML/DL 공통 평가 지표
│       ├── paths.py                  # 프로젝트 파일 경로 관리
│       ├── artifact_io.py            # ML/DL 모델 및 CSV 저장
│       ├── ml_training.py            # ML 분할·전처리·모델 로딩 보조 함수
│       └── model_promotion.py        # 튜닝 모델 리더보드 승격
│
└── uv.lock                           # uv를 통한 패키지 버전 및 의존성 고정
```

## 설치 및 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/)를 기준으로 합니다.

\`\`\`powershell
uv sync
uv run streamlit run main.py
\`\`\`

CLI는 저장소 루트에서 모듈 방식으로 실행합니다.

\`\`\`powershell
uv run python -m scripts.run_eda
uv run python -m scripts.train_ml
uv run python -m scripts.train_ml --models logistic_regression random_forest
uv run python -m scripts.evaluate_models
uv run python -m scripts.run_clustering --clusters 4
\`\`\`

테스트:

\`\`\`powershell
uv run pytest
\`\`\`

## 모델링 원칙

- 하나의 계층화 train/validation 분할과 random seed를 공통 사용
- 수치형: 중앙값 대치 후 표준화
- 범주형: 최빈값 대치 후 One-Hot Encoding
- Logistic Regression, Random Forest, XGBoost, LightGBM 기준선 비교
- ML 모델 정의는 모델별 Python 파일로 분리하고 trainer는 공통 학습 흐름만 담당
- ROC-AUC 우선, F1 보조 기준으로 최고 ML 모델 선정
- 튜닝, 임계값 최적화, 교차검증은 다음 단계에서 수행

## 머신러닝(ML) 공통 학습 및 협업 안내

### 사용 데이터

머신러닝 모델은 다음 전처리 완료 데이터를 공통으로 사용합니다.

```text
data/preprocessing/train_processed.csv
```

- 데이터 크기: 59,598행, 모델 입력 피처 41개
- CSV 저장 과정에서 생성된 `Unnamed: 0` 인덱스 열은 공통 학습 코드에서 제거
- 원핫 인코딩 결과인 `True/False` 값은 모델 입력용 `1/0`으로 변환
- 전처리 파일에는 `Left=0`, `Stayed=1`로 저장되어 있으나, 공통 학습 시
  퇴사를 양성 클래스로 평가하기 위해 `Left=1`, `Stayed=0`으로 변환


`train.py`는 개별 모델을 구현하는 파일이 아니라 팀원들이 만든 모델을 동일한 조건으로
학습하고 비교하는 공통 관리자입니다.

1. `train_processed.csv` 로드 및 입력값 검증
2. 타깃을 `Left=1`, `Stayed=0`으로 통일
3. 데이터를 학습 60%, 검증 20%, 최종 테스트 20%로 계층 분할
4. 구현 완료된 ML 모델의 생성 함수 호출
5. 모든 모델을 동일한 학습·검증 데이터로 평가
6. 검증 ROC-AUC 우선, F1 보조 기준으로 순위 결정
7. 최고 모델을 학습+검증 데이터로 재학습
8. 최종 테스트 데이터로 일반화 성능 평가
9. 성능표와 학습 모델 저장

### 모델 파일 구현 규약

각 모델 담당자는 원칙적으로 공통 `train.py`를 수정하지 않고 담당 모델 파일에 다음
이름의 생성 함수를 구현합니다.

| 담당 모델 | 파일 | 필수 생성 함수 |
|---|---|---|
| Logistic Regression | `logistic_regression.py` | `create_logistic_regression()` |
| Random Forest | `random_forest.py` | `create_random_forest()` |
| XGBoost | `xgboost.py` | `create_xgboost()` |
| LightGBM | `lightgbm.py` | `create_lightgbm()` |

생성 함수는 아직 학습되지 않은 scikit-learn 호환 분류기 또는 `Pipeline` 객체를
반환해야 합니다. 데이터 로드, 데이터 분할, 공통 평가, 모델 저장은 각 모델 파일에서
중복 구현하지 않고 `train.py`에 맡깁니다.

아직 구현되지 않았거나 필요한 라이브러리를 불러올 수 없는 모델은 자동으로 제외되므로
다른 모델의 학습을 막지 않습니다. 모델별 튜닝 코드는 `xgboost_tuning.py`처럼 별도
파일로 작성합니다. 딥러닝 모델은 `src/models/dl`의 별도 학습 흐름을 사용합니다.

### 실행 방법

구현된 모든 ML 모델을 동일한 조건으로 학습하고 비교합니다.

```powershell
uv run python -m src.models.ml.train
```

개발 중 특정 모델만 시험하고 기존 산출물을 덮어쓰지 않으려면 Python에서 다음처럼
호출합니다.

```python
from src.models.ml.train import run_training

leaderboard, final_metrics, unavailable = run_training(
    ["xgboost"],
    save_artifacts=False,
)
```

XGBoost 전체 하이퍼파라미터 튜닝은 다음 명령으로 실행합니다. 현재 설정은
`tree_method="hist"`, `device="cuda"` 방식으로 NVIDIA GPU를 사용합니다.

```powershell
uv run python -m src.models.ml.xgboost_tuning
```


### Git 협업 규칙

- `train.py`와 `utils.py`는 공통 담당자와 협의한 경우에만 수정합니다.
- 모델 담당자는 자신의 모델 파일과 별도 튜닝 파일만 수정합니다.
- 데이터 분할과 평가 기준을 모델 파일에서 임의로 변경하지 않습니다.
- 생성된 `.joblib` 모델과 결과 CSV는 개인 브랜치에서 직접 병합하지 않습니다.
- 전체 모델이 병합된 후 공통 담당자가 `train.py`를 실행해 최종 성능표와 최고 모델을
  생성합니다.
