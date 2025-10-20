import csv
import random
from datetime import datetime, timedelta

random.seed(42)

products = ['Green Tea', 'Black Tea', 'Oolong', 'Herbal Mix', 'Chamomile', 'Earl Grey', 'Matcha', 'Mint Tea', 'Chamomile Blend', 'Lemon Tea']
stores = ['Downtown', 'Uptown', 'Suburb', 'Mall', 'Airport']
staff = ['Alice', 'Bob', 'Chen', 'Diana', 'Evan']

headers = ['Date','OrderID','Product','Size','Price','Quantity','Total','Store','Staff','Notes']

rows = []
start = datetime(2023,1,1)

for i in range(900):
    d = start + timedelta(days=random.randint(0, 730))
    date_str = d.strftime(random.choice(['%Y-%m-%d','%d/%m/%Y','%b %d %Y','%d-%b-%Y']))
    prod = random.choice(products)
    size = random.choice(['S','M','L'])
    price = round(random.uniform(1.5,8.0) * (1 if size=='S' else 1.2 if size=='M' else 1.5),2)
    # introduce messy price strings sometimes
    if random.random() < 0.12:
        price_str = f"${price}"
    elif random.random() < 0.06:
        price_str = f"{price},00"
    else:
        price_str = str(price)
    qty = random.choice([1,1,1,2,2,3])
    # occasionally missing quantity
    if random.random() < 0.03:
        qty = ''
    total = '' if qty=='' else round(float(price)*int(qty),2)
    store = random.choice(stores)
    staff_member = random.choice(staff)
    notes = random.choice(['', 'promo', 'coupon', 'returned' if random.random()<0.02 else ''])
    order_id = f"ORD{100000 + i}"
    rows.append([date_str, order_id, prod, size, price_str, qty, total, store, staff_member, notes])

# add clear redundancies: duplicate blocks and corrupted rows
for i in range(100):
    # pick a base row and duplicate with small noise
    base = random.choice(rows[:300])
    new = base.copy()
    # sometimes blank Price or Total
    if random.random() < 0.2:
        new[4] = ''
    if random.random() < 0.2:
        new[6] = ''
    # messy product naming
    if random.random() < 0.15:
        new[2] = new[2].lower().replace(' ', '')
    rows.append(new)

# add exact duplicates to reach >100 redundancies
rows.extend(rows[:50])

# shuffle to mix duplicates
random.shuffle(rows)

# trim or pad to 1000 rows
rows = rows[:1000]

with open('c:\\Users\\ajayw\\Downloads\\sales_prediction_data_dashboard_v2\\teashop_dirty_1000.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print('teashop_dirty_1000.csv generated with', len(rows), 'rows')
