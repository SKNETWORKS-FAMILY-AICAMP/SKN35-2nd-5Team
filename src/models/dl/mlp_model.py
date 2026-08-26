import torch
import torch.nn as nn


# Optuna로 탐색된 최적 하이퍼파라미터로 MLP 구조를 생성하는 클래스
class MLPClassifier(nn.Module):
    """
    Optuna 파라미터 기반 Tabular Deep Learning 모델.
    Standard MLP 및 Residual Connection 구조를 모두 지원합니다.
    """

    def __init__(self, params: dict, in_features: int):
        super().__init__()
        self.params = params
        use_residual = params.get("use_residual", True)
        act_name = params.get("activation", "gelu")
        n_layers = params["n_layers"]

        if use_residual:
            hidden_dim = params.get("n_units_l0", 128)
            self.input_layer = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                get_activation(act_name),
            )
            blocks = []
            for i in range(n_layers):
                drop_rate = params.get(f"dropout_l{i}", params.get("dropout_l0", 0.2))
                blocks.append(ResidualBlock(hidden_dim, dropout=drop_rate, act_name=act_name))
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden_dim, 1)
        else:
            layers = []
            curr_in = in_features
            for i in range(n_layers):
                out_features = params[f"n_units_l{i}"]
                drop_rate = params.get(f"dropout_l{i}", 0.2)
                layers.append(nn.Linear(curr_in, out_features))
                layers.append(nn.BatchNorm1d(out_features))
                layers.append(get_activation(act_name))
                layers.append(nn.Dropout(drop_rate))
                curr_in = out_features
            self.input_layer = nn.Identity()
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(curr_in, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming Normal 초기화로 안정적인 초기 학습을 보장합니다."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.backbone(x)
        return self.head(x)


def get_activation(act_name: str) -> nn.Module:
    """선택된 문자열에 해당하는 활성화 함수 모듈을 반환합니다."""
    act_lower = str(act_name).lower()
    if act_lower == "gelu":
        return nn.GELU()
    elif act_lower == "silu":
        return nn.SiLU()
    elif act_lower == "leaky_relu":
        return nn.LeakyReLU(0.1)
    return nn.ReLU()


class ResidualBlock(nn.Module):
    """정형 데이터의 피처 보존 및 그래디언트 흐름을 극대화하는 Residual Block."""

    def __init__(self, dim: int, dropout: float = 0.2, act_name: str = "gelu"):
        super().__init__()
        act_fn = get_activation(act_name)
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
