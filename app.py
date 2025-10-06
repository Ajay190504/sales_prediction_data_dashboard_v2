from flask import Flask, render_template, request, jsonify, send_file, make_response
import pandas as pd, numpy as np, io, os, uuid, traceback
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

ALLOWED_EXT = {'.csv', '.xls', '.xlsx'}
UPLOAD_DIR = 'sessions'

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128 MB

def ensure_session_dir(sid):
    d = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d

def make_session_id():
    sid = request.cookies.get('SID')
    if sid:
        return sid
    sid = str(uuid.uuid4())
    return sid

def set_session_cookie(resp, sid):
    resp.set_cookie('SID', sid, samesite='Lax')
    return resp

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def read_file_to_df(filestream, filename):
    name = filename.lower()
    filestream.seek(0)
    if name.endswith('.csv'):
        return pd.read_csv(filestream)
    elif name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filestream)
    else:
        filestream.seek(0)
        return pd.read_csv(filestream)

def auto_clean(df: pd.DataFrame):
    df = df.copy()
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                if parsed.notna().mean() > 0.5:
                    df[col] = parsed
            except Exception:
                pass
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col].astype(str).str.replace(',','').str.strip(), errors='coerce')
            if coerced.notna().mean() > 0.5:
                df[col] = coerced
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)
    return df

def df_to_excel_bytes(df: pd.DataFrame):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    out.seek(0)
    return out

@app.route('/')
def index():
    sid = make_session_id()
    resp = make_response(render_template('index.html'))
    resp = set_session_cookie(resp, sid)
    return resp

@app.route('/api/session-status', methods=['GET'])
def session_status():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    cleaned_path = os.path.join(d, 'cleaned.csv')
    preds_path = os.path.join(d, 'predictions.csv')
    status = {'has_raw': False, 'has_cleaned': False, 'has_predictions': False, 'columns': []}
    raw_path = os.path.join(d, 'raw.csv')
    if os.path.exists(raw_path):
        status['has_raw'] = True
        try:
            df = pd.read_csv(raw_path)
            status['columns'] = list(df.columns)
        except Exception:
            pass
    if os.path.exists(cleaned_path):
        status['has_cleaned'] = True
        try:
            dfc = pd.read_csv(cleaned_path)
            status['columns'] = list(dfc.columns)
        except Exception:
            pass
    if os.path.exists(preds_path):
        status['has_predictions'] = True
    return jsonify(status)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    try:
        df = read_file_to_df(file.stream, file.filename)
    except Exception as e:
        return jsonify({'error': 'Failed to read file', 'detail': str(e)}), 400
    raw_path = os.path.join(d, 'raw.csv')
    df.to_csv(raw_path, index=False)
    preview = df.head(500).to_dict(orient='records')
    return jsonify({'columns': list(df.columns), 'preview': preview})

@app.route('/api/clean', methods=['POST'])
def api_clean():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    raw_path = os.path.join(d, 'raw.csv')
    if not os.path.exists(raw_path):
        return jsonify({'error': 'No raw file uploaded'}), 400
    try:
        df = pd.read_csv(raw_path)
        cleaned = auto_clean(df)
        cleaned_path = os.path.join(d, 'cleaned.csv')
        cleaned.to_csv(cleaned_path, index=False)
        preview = cleaned.head(1000).to_dict(orient='records')
        return jsonify({'columns': list(cleaned.columns), 'preview': preview})
    except Exception as e:
        return jsonify({'error': 'Cleaning failed', 'detail': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/download/cleaned', methods=['GET'])
def download_cleaned():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    cleaned_path = os.path.join(d, 'cleaned.csv')
    if not os.path.exists(cleaned_path):
        return jsonify({'error': 'No cleaned data'}), 400
    return send_file(cleaned_path, download_name='cleaned_data.csv', as_attachment=True, mimetype='text/csv')

@app.route('/api/get-cleaned', methods=['GET'])
def api_get_cleaned():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    cleaned_path = os.path.join(d, 'cleaned.csv')
    if not os.path.exists(cleaned_path):
        return jsonify({'error': 'No cleaned data'}), 400
    df = pd.read_csv(cleaned_path)
    preview = df.head(2000).to_dict(orient='records')
    return jsonify({'columns': list(df.columns), 'preview': preview, 'rows': len(df)})

@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    cleaned_path = os.path.join(d, 'cleaned.csv')
    raw_path = os.path.join(d, 'raw.csv')
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        return jsonify({'error': 'No data available'}), 400
    payload = request.json or {}
    kind = payload.get('kind')
    x = payload.get('x')
    y = payload.get('y')
    z = payload.get('z')  # for heatmap
    groupby = payload.get('groupby')
    try:
        if kind == 'pie':
            if y:
                series = df.groupby(x)[y].sum() if x else df[y].value_counts()
                labels = series.index.astype(str).tolist()
                values = series.values.tolist()
                return jsonify({'type':'pie','labels':labels,'values':values,'layout':{'title':payload.get('title','Pie Chart')}})
            else:
                return jsonify({'error':'Pie requires y (values) or x for counts'}),400
        elif kind == 'heatmap' and x and y:
            pivot = pd.pivot_table(df, values=z or y, index=y, columns=x, aggfunc='mean', fill_value=0)
            zvals = pivot.values.tolist()
            xlabels = list(pivot.columns.astype(str))
            ylabels = list(pivot.index.astype(str))
            return jsonify({'type':'heatmap','x':xlabels,'y':ylabels,'z':zvals,'layout':{'title':payload.get('title','Heatmap')}})
        elif kind == 'corr':
            corr = df.select_dtypes(include=[np.number]).corr().round(3)
            mat = corr.values.tolist()
            cols = list(corr.columns)
            return jsonify({'type':'corr','cols':cols,'matrix':mat,'layout':{'title':payload.get('title','Correlation Matrix')}})
        else:
            if kind in ('line','bar','scatter','area'):
                xvals = df[x].astype(str).tolist() if x else list(range(len(df)))
                if isinstance(y, list):
                    series = []
                    for yy in y:
                        series.append({'name': yy, 'x': xvals, 'y': df[yy].tolist()})
                    return jsonify({'type':'multi','series':series,'layout':{'title':payload.get('title','Multi Series')}})
                else:
                    yvals = df[y].tolist() if y else []
                    return jsonify({'type':kind,'x':xvals,'y':yvals,'layout':{'title':payload.get('title','Chart')}})
            elif kind == 'histogram':
                xvals = df[x].dropna().tolist()
                return jsonify({'type':'histogram','x':xvals,'layout':{'title':payload.get('title','Histogram')}})
            elif kind == 'box':
                yvals = df[y].dropna().tolist()
                return jsonify({'type':'box','y':yvals,'layout':{'title':payload.get('title','Box Plot')}})
            else:
                return jsonify({'error':'Unsupported chart type'}),400
    except Exception as e:
        return jsonify({'error':'Visualization failed','detail':str(e),'trace':traceback.format_exc()}),500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    cleaned_path = os.path.join(d, 'cleaned.csv')
    raw_path = os.path.join(d, 'raw.csv')
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        return jsonify({'error':'No data available'}),400
    payload = request.json or {}
    time_col = payload.get('time_column')
    target = payload.get('target')
    model_name = payload.get('model','random_forest')
    horizon = int(payload.get('horizon',6))
    try:
        if target not in df.columns:
            return jsonify({'error':'Target column not in data'}),400
        df_model = df.copy()
        if time_col and time_col in df_model.columns:
            df_model[time_col] = pd.to_datetime(df_model[time_col], errors='coerce')
            df_model = df_model.sort_values(time_col).reset_index(drop=True)
        df_model[target] = pd.to_numeric(df_model[target].astype(str).str.replace(',',''), errors='coerce')
        def make_lag(series,n=3):
            tmp = pd.DataFrame({'y':series})
            for i in range(1,n+1):
                tmp[f'lag_{i}'] = tmp['y'].shift(i)
            tmp.dropna(inplace=True)
            return tmp
        lag = make_lag(df_model[target], n=3)
        if lag.empty:
            return jsonify({'error':'Not enough data to build lags (>3 rows required)'}),400
        df2 = df_model.iloc[len(df_model)-len(lag):].reset_index(drop=True)
        for c in lag.columns:
            if c!='y':
                df2[c]=lag[c].values
        features = [c for c in df2.columns if str(c).startswith('lag_')]
        X = df2[features].astype(float).fillna(0)
        y = df2[target].astype(float).fillna(0)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
        if model_name=='xgboost' and XGBOOST_AVAILABLE:
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train,y_train)
        ypred_test = model.predict(X_test)
        mse = float(mean_squared_error(y_test, ypred_test))
        r2 = float(r2_score(y_test, ypred_test))
        recent = X.iloc[-1].values.astype(float)
        future = []
        future_idx = []
        last_time = None
        if time_col and time_col in df_model.columns and np.issubdtype(df_model[time_col].dtype, np.datetime64):
            last_time = pd.to_datetime(df_model[time_col].iloc[-1])
        for h in range(horizon):
            x_in = recent.reshape(1,-1)
            p = float(model.predict(x_in)[0])
            future.append(p)
            if last_time is not None:
                future_idx.append((last_time + pd.DateOffset(months=h+1)).strftime('%Y-%m-%d'))
            recent = np.roll(recent,1)
            recent[0]=p
        preds_df = pd.DataFrame({f'{target}_predicted': future})
        if future_idx:
            preds_df['period'] = future_idx
        preds_path = os.path.join(d, 'predictions.csv')
        preds_df.to_csv(preds_path, index=False)
        excel_bytes = df_to_excel_bytes(preds_df)
        with open(os.path.join(d,'predictions.xlsx'),'wb') as f:
            f.write(excel_bytes.getvalue())
        return jsonify({'metrics':{'mse':mse,'r2':r2}, 'predictions_preview': preds_df.head(500).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error':'Prediction failed','detail':str(e),'trace':traceback.format_exc()}),500

@app.route('/download/predictions', methods=['GET'])
def download_predictions():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    preds_csv = os.path.join(d,'predictions.csv')
    if not os.path.exists(preds_csv):
        return jsonify({'error':'No predictions available'}),400
    return send_file(preds_csv, download_name='predictions.csv', as_attachment=True, mimetype='text/csv')

if __name__=='__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
