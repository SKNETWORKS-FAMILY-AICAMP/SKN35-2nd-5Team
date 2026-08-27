"""로지스틱 회귀 모델 구현 파일.

데이터 로딩·분할·평가·저장은 ``train.py``가 공통으로 처리한다.
이 파일에서는 아래 import를 사용해 ``create_logistic_regression()``만 구현하면 된다.
"""

# 로지스틱 회귀는 피처 크기에 영향을 받으므로 StandardScaler를 Pipeline으로 묶어 사용한다.
from sklearn.linear_model import LogisticRegression  
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  

from .utils import RANDOM_STATE  

def create_logistic_regression() -> Pipeline:
    """StandardScaler와 LogisticRegression을 결합한 Pipeline 객체를 생성

    `train.py`에서 공통 전처리기(결측치 처리, 원핫 인코딩) 뒤에 본 파이프라인이
    연결, 스케일링(StandardScaler) 후 로지스틱 회귀를 수행하도록 구성
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,  # 수렴(convergence) 경고 방지를 위한 반복 횟수
                ),
            ),
        ]
    )
    return model
