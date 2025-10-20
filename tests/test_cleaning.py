import os
import pandas as pd
from app import auto_clean

BASE = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(BASE, 'teashop_dirty_1000.csv')

def test_auto_clean_removes_duplicates_and_fills():
    df = pd.read_csv(SRC)
    before_dupes = df.duplicated().sum()
    cleaned = auto_clean(df)
    after_dupes = cleaned.duplicated().sum()
    # expect duplicates removed
    assert after_dupes == 0
    # numeric columns should have no NaNs
    numeric_cols = cleaned.select_dtypes(include=['number']).columns
    for c in numeric_cols:
        assert cleaned[c].isna().sum() == 0
    # expect at least some duplicates were present in source
    assert before_dupes >= 50
