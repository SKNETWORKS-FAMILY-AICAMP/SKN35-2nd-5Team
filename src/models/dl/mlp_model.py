import torch.nn as nn


# Optuna로 탐색된 최적 하이퍼파라미터로 MLP 구조를 생성하는 클래스
class MLPClassifier(nn.Module):
    def __init__(self, best_params, in_features):
        super().__init__()
        layers = []
        n_layers = best_params["n_layers"]

        for i in range(n_layers):
            out_features = best_params[f"n_units_l{i}"]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(best_params[f"dropout_l{i}"]))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_extractor(x)
