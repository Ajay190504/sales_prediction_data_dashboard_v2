import csv
import random
from datetime import datetime, timedelta

random.seed(2025)

# Define 20 columns for AWS-like billing/usage
columns = [
    'Date','Account','Region','Service','ResourceId','UsageType','UsageAmount','UsageUnit','Cost','Currency',
    'Operation','API','AvailabilityZone','PurchaseOption','MeterId','Project','Environment','Tags','BillingPeriod','InvoiceId'
]

services = ['EC2','S3','RDS','Lambda','EKS','CloudFront','DynamoDB','SNS','SQS','ECR']
regions = ['us-east-1','us-west-2','eu-west-1','ap-south-1','ap-northeast-1']
accounts = ['acct-1001','acct-1002','acct-1003']
usage_types = ['DataTransfer-Out','BoxUsage','TimedStorage-ByteHrs','Requests','GB-Months']
ops = ['RunInstances','GetObject','PutObject','Invoke','DescribeInstances']
azs = ['us-east-1a','us-east-1b','us-west-2a','eu-west-1a']
projects = ['projA','projB','projC']
envs = ['prod','staging','dev']

rows = []
start = datetime(2025,1,1)

for i in range(180):
    d = start + timedelta(days=random.randint(0, 300))
    row = {}
    row['Date'] = d.strftime('%Y-%m-%d')
    row['Account'] = random.choice(accounts)
    row['Region'] = random.choice(regions)
    svc = random.choice(services)
    row['Service'] = svc
    row['ResourceId'] = f"{svc.lower()}-{random.randint(1000,9999)}"
    row['UsageType'] = random.choice(usage_types)
    # usage sometimes missing
    row['UsageAmount'] = round(random.uniform(0.01, 1000.0),3) if random.random() > 0.02 else ''
    row['UsageUnit'] = 'hours' if svc=='EC2' else 'GB' if svc in ('S3','ECR') else 'requests'
    # cost sometimes messy
    cost = round(float(row['UsageAmount'] or 0) * random.uniform(0.001, 0.2),4) if row['UsageAmount']!='' else ''
    if cost != '' and random.random() < 0.1:
        row['Cost'] = f"${cost}"
    else:
        row['Cost'] = str(cost)
    row['Currency'] = 'USD'
    row['Operation'] = random.choice(ops)
    row['API'] = random.choice(['REST','SDK','CLI'])
    row['AvailabilityZone'] = random.choice(azs)
    row['PurchaseOption'] = random.choice(['OnDemand','Reserved','Spot'])
    row['MeterId'] = f"MID{random.randint(10000,99999)}"
    row['Project'] = random.choice(projects)
    row['Environment'] = random.choice(envs)
    row['Tags'] = random.choice(['', 'team:alpha', 'owner:devops', 'costcenter:42'])
    row['BillingPeriod'] = d.strftime('%Y-%m')
    row['InvoiceId'] = f"INV{random.randint(100000,999999)}"
    rows.append([row[c] for c in columns])

# add 20 rows with intentional missing fields and duplicates
for i in range(20):
    base = random.choice(rows[:100])
    new = base.copy()
    if random.random() < 0.4:
        # blank cost
        new[8] = ''
    if random.random() < 0.3:
        new[6] = ''
    rows.append(new)

# add exact duplicates
rows.extend(rows[:10])
random.shuffle(rows)
rows = rows[:200]

out = 'c:\\\\Users\\\\ajayw\\\\Downloads\\\\sales_prediction_data_dashboard_v2\\\\aws_usage_200_20cols.csv'
with open(out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)
print('Generated', out, 'with', len(rows), 'rows and', len(columns), 'columns')
