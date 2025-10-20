import os
import pandas as pd
from app import auto_clean

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'teashop_dirty_1000.csv')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sessions')
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'test_cleaned.csv')

print('Reading:', SRC)
df = pd.read_csv(SRC)
print('Rows before:', len(df))
num_cols = df.select_dtypes(include=['number']).columns.tolist()
# sometimes numeric columns may be object; coerce potential numeric-like columns
possible_numeric = [c for c in df.columns if df[c].astype(str).str.replace('[^0-9.+-]', '', regex=True).str.len().gt(0).any()]

print('Columns:', list(df.columns))
print('Duplicate rows before:', df.duplicated().sum())

# count numeric NaNs before (after coercion where needed)
numeric_before = {}
for c in df.columns:
    coerced = pd.to_numeric(df[c].astype(str).str.replace('[,$]', '', regex=True).str.replace(',',''), errors='coerce')
    numeric_before[c] = int(coerced.isna().sum())
print('Numeric-like NaNs before (per column):')
for k,v in numeric_before.items():
    print(f'  {k}: {v}')

cleaned = auto_clean(df)
print('\nAfter auto_clean:')
print('Rows after:', len(cleaned))
print('Duplicate rows after:', cleaned.duplicated().sum())

# numeric NaNs after
numeric_after = {}
for c in cleaned.columns:
    if pd.api.types.is_numeric_dtype(cleaned[c]):
        numeric_after[c] = int(cleaned[c].isna().sum())
print('Numeric NaNs after (per numeric column):')
for k,v in numeric_after.items():
    print(f'  {k}: {v}')

# save cleaned output
cleaned.to_csv(OUT, index=False)
print('\nCleaned CSV saved to', OUT)
print('\nPreview (first 5 rows):')
print(cleaned.head().to_string(index=False))
