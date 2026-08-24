"""Print the saved model leaderboard."""

import pandas as pd

from src.utils.paths import DL_METRICS_PATH, ML_LEADERBOARD_PATH


def main() -> None:
    if not ML_LEADERBOARD_PATH.exists():
        raise FileNotFoundError("리더보드가 없습니다. 먼저 train_ml을 실행하세요.")
    leaderboard = pd.read_csv(ML_LEADERBOARD_PATH)
    if DL_METRICS_PATH.exists():
        dl_metrics = pd.read_csv(DL_METRICS_PATH)
        common = [column for column in leaderboard.columns if column in dl_metrics.columns]
        leaderboard = pd.concat([leaderboard, dl_metrics[common]], ignore_index=True)
    leaderboard = leaderboard.sort_values(["roc_auc", "f1"], ascending=False)
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
