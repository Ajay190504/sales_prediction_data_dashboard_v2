import os, json
from app import app, ensure_session_dir

# Use test client to POST to /api/predict using an existing cleaned file
base = os.path.dirname(os.path.dirname(__file__))
sessions_dir = os.path.join(base, 'sessions')
# pick the cleaned file created earlier
cleaned_src = os.path.join(sessions_dir, 'test_cleaned.csv')
if not os.path.exists(cleaned_src):
    print('Cleaned file not found at', cleaned_src)
    raise SystemExit(1)

# create a fresh session id folder
sid = 'predict_test_session'
d = ensure_session_dir(sid)
# copy cleaned file into that session folder as cleaned.csv
import shutil
shutil.copy(cleaned_src, os.path.join(d, 'cleaned.csv'))

print('Session folder:', d)
print('cleaned.csv exists?', os.path.exists(os.path.join(d,'cleaned.csv')))

client = app.test_client()
# set SID cookie on client; try common signatures for different Werkzeug versions
try:
    client.set_cookie('SID', sid)
except TypeError:
    try:
        client.set_cookie('localhost', 'SID', sid)
    except Exception:
        # last resort: pass cookie via headers when calling endpoints
        pass

payload = {
    'time_column': 'Date',
    'target': 'Total',
    'model': 'random_forest',
    'horizon': 6,
    'freq': 'auto',
    'plot_type': 'none',
    'ml_model': 'random_forest'
}
print('\nCalling /api/session-status to see server session awareness...')
resp_status = client.get('/api/session-status')
print('session-status:', resp_status.status_code, resp_status.get_json())

print('\nNow calling /api/predict')
resp = client.post('/api/predict', json=payload)
print('Status code:', resp.status_code)
try:
    data = resp.get_json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print('Failed to decode JSON response:', e)
    print(resp.data.decode('utf-8'))
