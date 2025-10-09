from app import app
from pathlib import Path
import shutil, json

methods = ['sum','mean','last']
client = app.test_client()
# start new session
r = client.post('/api/new-session')
SID = r.get_json().get('sid')
print('SID:', SID)
# copy demo file
p = Path('sessions')/SID
p.mkdir(parents=True, exist_ok=True)
shutil.copy('demo_sales.csv', p/'raw.csv')
# clean
r2 = client.post('/api/clean', json={'force': True})
print('clean:', r2.status_code, r2.get_json() and r2.get_json().get('already_cleaned'))

for m in methods:
    payload = {'time_column':'Date','target':'Sales','model':'random_forest','ml_model':'random_forest','freq':'M','plot_type':'plotly','compare':False,'horizon':6,'aggregate':True,'aggregate_method':m}
    res = client.post('/api/predict', json=payload)
    print('\nmethod:', m, 'status:', res.status_code)
    try:
        print(json.dumps(res.get_json(), indent=2)[:2000])
    except Exception as e:
        print('failed to parse json', e)

print('\nsession files:', list(p.iterdir()))
