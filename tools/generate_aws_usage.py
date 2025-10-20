import csv
import random
from datetime import datetime, timedelta

random.seed(123)
services = ['EC2','S3','RDS','Lambda','EKS','CloudFront','DynamoDB','SNS','SQS']
regions = ['us-east-1','us-west-2','eu-west-1','ap-south-1','ap-northeast-1']
accounts = ['acct-1001','acct-1002','acct-1003']
headers = ['Date','Account','Region','Service','UsageAmount','UsageUnit','Cost','ResourceId']
rows = []
start = datetime(2025,1,1)
for i in range(450):
    d = start + timedelta(days=random.randint(0, 300))
    date_str = d.strftime('%Y-%m-%d')
    acct = random.choice(accounts)
    region = random.choice(regions)
    svc = random.choice(services)
    usage = round(random.uniform(0.01, 500.0),3)
    unit = 'hours' if svc=='EC2' else 'GB' if svc in ('S3','EBS') else 'requests'
    # messy cost formatting sometimes
    cost = round(usage * random.uniform(0.001, 0.2), 4)
    if random.random() < 0.08:
        cost_str = f"${cost}"
    else:
        cost_str = str(cost)
    resource = f"{svc.lower()}-{random.randint(1000,9999)}"
    rows.append([date_str, acct, region, svc, usage, unit, cost_str, resource])

# add duplicates and missing
for i in range(30):
    base = random.choice(rows)
    new = base.copy()
    if random.random() < 0.3:
        new[6] = ''  # missing cost
    if random.random() < 0.2:
        new[4] = ''  # missing usage
    rows.append(new)

# exact duplicates to ensure cleaning needs to remove them
rows.extend(rows[:20])

random.shuffle(rows)
rows = rows[:500]

with open('c:\\\\Users\\\\ajayw\\\\Downloads\\\\sales_prediction_data_dashboard_v2\\\\aws_usage_500.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)
print('aws_usage_500.csv generated with', len(rows), 'rows')
