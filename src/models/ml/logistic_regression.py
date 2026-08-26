"""로지스틱 회귀 모델 구현 파일.

데이터 로딩·분할·평가·저장은 ``train.py``가 공통으로 처리한다.
이 파일에서는 아래 import를 사용해 ``create_logistic_regression()``만 구현하면 된다.
"""

# 로지스틱 회귀는 피처 크기에 영향을 받으므로 StandardScaler를 Pipeline으로 묶어 사용한다.
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

from .utils import RANDOM_STATE  # noqa: F401
