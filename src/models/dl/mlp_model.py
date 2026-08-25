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


# Optuna trial 객체를 받아 탐색용 임시 MLP 구조를 생성하는 함수
# def build_trial_model(trial, in_features):
#     n_layers = trial.suggest_int("n_layers", 1, 3)
#     layers = []

#     for i in range(n_layers):
#         out_features = trial.suggest_int(f"n_units_l{i}", 16, 128)
#         layers.append(nn.Linear(in_features, out_features))
#         layers.append(nn.BatchNorm1d(out_features))
#         layers.append(nn.ReLU())

#         dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
#         layers.append(nn.Dropout(dropout_rate))
#         in_features = out_features

#     layers.append(nn.Linear(in_features, 1))
#     return nn.Sequential(*layers)
