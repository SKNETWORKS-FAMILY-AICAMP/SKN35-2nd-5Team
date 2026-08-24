import pandas as pd
import pytest

from src.load_data.loader import load_dataset, split_features_target


def test_load_dataset_and_split(tmp_path, sample_frame):
    path = tmp_path / "employees.csv"
    sample_frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    features, target = split_features_target(loaded)
    assert len(loaded) == len(sample_frame)
    assert "Employee ID" not in features.columns
    assert "Attrition" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_duplicate_employee_id_is_rejected(tmp_path, sample_frame):
    duplicated = pd.concat([sample_frame, sample_frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicated.csv"
    duplicated.to_csv(path, index=False)
    with pytest.raises(ValueError, match="중복"):
        load_dataset(path)
