import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(2025)

rows = []
products = ['T-Shirt', 'TShirt', 'tshirt ', 'Jeans', 'jeans', 'Jacket']
sizes = ['S', 'M', 'L', 'XL']
channels = ['Online', 'Retail']
regions = ['North','South','East','West']

dates = ['2024-01-05','05/02/2024','2024-03-01','2024-03-01','2024-04-01','2024-04-15','2024-05-01']

for d in dates:
    for p in products:
        # create duplicates intentionally
        for dup in range(np.random.choice([1,1,2])):
            units = np.random.choice([5,10,15,20, np.nan])
            price = np.random.choice(["$10.00","10","10,00"," 12.5 ", None])
            revenue = None
            # sometimes revenue is provided incorrectly
            if price and units and not pd.isna(units):
                try:
                    pr = float(str(price).replace('$','').replace(',','.') )
                    revenue = round(pr * units + np.random.normal(scale=2.0),2)
                except Exception:
                    revenue = None
            rows.append({
                'date': d,
                'product': p,
                'size': np.random.choice(sizes),
                'region': np.random.choice(regions),
                'channel': np.random.choice(channels),
                'units_sold': units,
                'price': price,
                'revenue': revenue
            })

# introduce a row with swapped columns and missing headers style (simulate messy export)
rows.append({'date':'2024/06/01','product':'Denim','size':'M','region':'North','channel':'Retail','units_sold':'', 'price':'$20','revenue':''})

# create dataframe and save
df = pd.DataFrame(rows)
# shuffle rows
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
# save
df.to_csv('demo_clothshop.csv', index=False)
print('demo_clothshop.csv created with', len(df), 'rows')
