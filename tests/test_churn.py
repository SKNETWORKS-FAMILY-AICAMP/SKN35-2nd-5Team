from src.analysis.retention import score_retention_risk
from src.ml.logistic_regression import create_logistic_regression
from src.ml.random_forest import create_random_forest
from src.ml.trainer import train_ml_models


def test_model_factories_are_independent():
    logistic = create_logistic_regression(random_state=7)
    forest = create_random_forest(random_state=7)
    assert logistic.random_state == 7
    assert forest.random_state == 7


def test_logistic_regression_training_and_scoring(sample_frame):
    results, leaderboard, unavailable = train_ml_models(
        sample_frame,
        selected=["logistic_regression"],
        test_size=0.25,
        save_artifacts=False,
    )
    assert not unavailable
    assert leaderboard.iloc[0]["model"] == "logistic_regression"
    assert 0.0 <= leaderboard.iloc[0]["roc_auc"] <= 1.0

    scored = score_retention_risk(results[0].pipeline, sample_frame)
    assert len(scored) == len(sample_frame)
    assert scored["attrition_probability"].between(0, 1).all()
    assert scored["recommended_action"].str.len().gt(0).all()
