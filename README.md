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

\`\`\`text
.
├── README.md # 프로젝트 설명 및 실행 방법
│
├── artifacts/ # 학습 결과물 저장
│ ├── dl/ # 딥러닝 모델 결과물
│ │ ├── mlp_best_params.pkl # MLP 최적 하이퍼파라미터
│ │ ├── mlp_model.pt # 학습된 MLP 모델
│ │ └── mlp_scaler.pkl # MLP 전처리에 사용한 Scaler
│ │
│ └── ml/ # 머신러닝 모델 결과물
│
├── data/ # 프로젝트 데이터
│ ├── preprocessing/ # 전처리된 데이터
│ │ └── train_processed.csv # 전처리 완료된 학습 데이터
│ │
│ └── raw/ # 원본 데이터
│ ├── test.csv # 테스트용 원본 데이터
│ └── train.csv # 학습용 원본 데이터
│
├── main.py # Streamlit 애플리케이션 실행 진입점
│
├── notebooks/ # 데이터 분석 및 모델 실험용 Jupyter Notebook
│ ├── dl/ # 딥러닝 실험
│ │ └── mlp.ipynb # MLP 모델 실험 및 학습 과정
│ │
│ ├── ml/ # 머신러닝 실험
│ │
│ └── preprocessing/ # 데이터 전처리 실험
│ └── processing.ipynb # 데이터 탐색 및 전처리 과정
│
├── pyproject.toml # Python 프로젝트 설정 및 의존성 관리
│
├── src/ # 실제 프로젝트에서 사용하는 소스 코드
│ ├── **init**.py # Python 패키지 초기화
│ ├── config.py # 프로젝트 전반의 설정값 관리
│ │
│ ├── data/ # 데이터 처리 관련 코드
│ │ ├── loader.py # 데이터 로딩
│ │ └── preprocess.py # 데이터 전처리 로직
│ │
│ ├── models/ # 머신러닝 / 딥러닝 모델 관련 코드
│ │ ├── dl/ # 딥러닝
│ │ │ ├── mlp_model.py # MLP 모델 구조 정의
│ │ │ └── train.py # 딥러닝 모델 학습 및 저장
│ │ │
│ │ └── ml/ # 머신러닝 모델
│ │
│ ├── pages/ # Streamlit 페이지
│ │ ├── 01_EDA.py # 데이터 탐색 및 EDA 화면
│ │ ├── 02_ML_Training.py # 머신러닝 학습 화면
│ │ ├── 03_Model_Comparison.py # 모델 성능 비교 화면
│ │ ├── 08_SHAP_Analysis.py # SHAP 기반 모델 해석 화면
│ │ └── 09_Retention_Action.py # 이탈 예측 결과 및 대응 전략 화면
│ │
│ └── utils/ # 여러 곳에서 공통으로 사용하는 기능
│ ├── **init**.py # Python 패키지 초기화
│ ├── constants.py # 공통 상수 및 설정값
│ ├── metrics.py # 모델 평가 지표 관련 함수
│ └── paths.py # 프로젝트 파일 경로 관리
│
└── uv.lock # uv를 통한 패키지 버전 및 의존성 고정
\`\`\`

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

