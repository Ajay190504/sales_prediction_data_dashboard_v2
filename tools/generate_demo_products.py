import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

start = pd.to_datetime('2019-01-01')
periods = 72  # 6 years monthly
products = ['Alpha', 'Beta', 'Gamma']
regions = ['North', 'South', 'East', 'West']
channels = ['Online', 'Retail']

rows = []
for i in range(periods):
    date = (start + pd.DateOffset(months=i)).strftime('%Y-%m-%d')
    for p in products:
        base = 200 + (products.index(p) * 50)  # different base per product
        season = 20 * np.sin(2 * np.pi * (i % 12) / 12)  # yearly seasonality
        trend = 2.5 * i  # gentle upward trend
        noise = np.random.normal(scale=15)
        units = max(1, int(base + season + trend + noise))
        price = round(10 + products.index(p) * 5 + np.random.normal(scale=0.5), 2)
        revenue = round(units * price, 2)
        cost = round(units * (price * 0.6 + np.random.normal(scale=0.2)), 2)
        rows.append({
            'date': date,
            'product': p,
            'region': np.random.choice(regions),
            'channel': np.random.choice(channels, p=[0.6, 0.4]),
            'units_sold': units,
            'price': price,
            'revenue': revenue,
            'cost': cost
        })

df = pd.DataFrame(rows)
# save to repo root
df.to_csv('demo_sales_products.csv', index=False)
print('demo_sales_products.csv generated with', len(df), 'rows')
