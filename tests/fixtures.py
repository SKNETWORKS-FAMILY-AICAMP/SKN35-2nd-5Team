"""Shared synthetic-data fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "Employee ID": range(1000, 1000 + rows),
            "Age": [22 + index % 35 for index in range(rows)],
            "Gender": ["Female" if index % 2 else "Male" for index in range(rows)],
            "Monthly Income": [3000 + index * 120 for index in range(rows)],
            "Overtime": ["Yes" if index % 3 == 0 else "No" for index in range(rows)],
            "Job Satisfaction": ["Low" if index % 4 == 0 else "High" for index in range(rows)],
            "Work-Life Balance": ["Poor" if index % 5 == 0 else "Good" for index in range(rows)],
            "Employee Recognition": ["Low" if index % 3 == 0 else "High" for index in range(rows)],
            "Leadership Opportunities": ["No" if index % 2 else "Yes" for index in range(rows)],
            "Attrition": ["Left" if index % 2 else "Stayed" for index in range(rows)],
        }
    )
