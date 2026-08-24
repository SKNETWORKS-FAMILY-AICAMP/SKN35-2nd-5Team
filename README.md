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
├─ main.py
├─ data/
├─ src/
│  ├─ load_data/     # CSV 로딩 및 검증
│  ├─ analysis/      # EDA 및 리텐션 제안
│  ├─ ml/            # 모델별 독립 파일, 전처리, 평가, SHAP, 군집화
│  └─ utils/         # 상수 및 경로
├─ pages/            # Streamlit 멀티페이지
├─ scripts/          # 일괄 실행 CLI
├─ tests/            # 단위 테스트
└─ artifacts/        # 실행 시 생성되는 모델 및 리포트
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

## 주의

이탈 확률과 리텐션 액션은 의사결정 지원용입니다. 중요도는 인과관계를 의미하지
않으며, 모델 결과만으로 채용·평가·보상·징계 등 자동 인사결정을 내려서는 안 됩니다.
