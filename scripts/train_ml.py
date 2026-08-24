import argparse

from src.load_data.loader import load_train_data
from src.ml.trainer import train_ml_models
from src.utils.constants import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="ML 기준선 일괄 학습")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    args = parser.parse_args()
    _, leaderboard, unavailable = train_ml_models(load_train_data(), selected=args.models)
    print(leaderboard.to_string(index=False))
    if unavailable:
        print(f"Unavailable: {unavailable}")


if __name__ == "__main__":
    main()
