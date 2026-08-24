"""Train the MLP baseline."""

import argparse
import json

from src.dl.mlp import train_mlp
from src.load_data.loader import load_train_data


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP 기준선 학습")
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    _, metrics = train_mlp(
        load_train_data(),
        hidden_layer_sizes=tuple(args.hidden_layers),
        max_iter=args.max_iter,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
