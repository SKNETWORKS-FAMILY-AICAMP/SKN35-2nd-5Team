"""Generate EDA report files."""

import json

from src.analysis.eda import build_eda_report, categorical_summary, numeric_summary
from src.load_data.loader import load_train_data
from src.utils.paths import REPORTS_DIR, ensure_artifact_dirs


def main() -> None:
    data = load_train_data()
    report = build_eda_report(data)
    ensure_artifact_dirs()
    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    numeric_summary(data).to_csv(REPORTS_DIR / "numeric_summary.csv", index=False)
    categorical_summary(data).to_csv(REPORTS_DIR / "categorical_summary.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
