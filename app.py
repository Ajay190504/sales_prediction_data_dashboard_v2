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


import base64
from io import BytesIO
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None

def make_plot_image(history_df=None, preds_df=None, title='Forecast'):
    if not MATPLOTLIB_AVAILABLE:
        return None
    fig, ax = plt.subplots(figsize=(8,4))
    try:
        if history_df is not None and len(history_df.columns) >= 2:
            hx = pd.to_datetime(history_df.iloc[:,0], errors='coerce')
            hy = pd.to_numeric(history_df.iloc[:,1].astype(str).str.replace(',',''), errors='coerce')
            ax.plot(hx, hy, label='History')
    except Exception:
        pass
    try:
        if preds_df is not None and not preds_df.empty:
            py = preds_df.iloc[:,0].astype(float).values
            px = list(range(len(py)))
            ax.plot(px, py, linestyle='--', marker='o', label='Forecast')
            if 'period' in preds_df.columns:
                ax.set_xticks(px)
                ax.set_xticklabels(preds_df['period'].astype(str), rotation=45, ha='right')
    except Exception:
        pass
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


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
    try:
        # openpyxl may not be installed in minimal environments
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        out.seek(0)
        return out
    except Exception:
        return None

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
    # Prefer cleaned file when reporting columns (so reload shows cleaned view after cleaning)
    raw_path = os.path.join(d, 'raw.csv')
    if os.path.exists(cleaned_path):
        status['has_cleaned'] = True
        try:
            dfc = pd.read_csv(cleaned_path)
            status['columns'] = list(dfc.columns)
        except Exception:
            # fallback to raw if cleaned cannot be read
            pass
    if os.path.exists(raw_path) and not status['columns']:
        status['has_raw'] = True
        try:
            df = pd.read_csv(raw_path)
            status['columns'] = list(df.columns)
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
        # If reading fails, return a JSON error so client can display a message
        return jsonify({'error': 'Failed to read file', 'detail': str(e)}), 400
    raw_path = os.path.join(d, 'raw.csv')
    df.to_csv(raw_path, index=False)
    # Build preview/columns safely — if this fails, the file has already been saved
    try:
        cols = [str(c) for c in df.columns]
        preview = df.head(500).to_dict(orient='records')
    except Exception:
        # Log exception server-side and return a minimal successful payload
        app.logger.exception('Failed to build preview after upload')
        cols = []
        preview = []
    resp = jsonify({'columns': cols, 'preview': preview, 'sid': sid})
    # ensure client has correct session cookie pointing to where the file was saved
    resp = set_session_cookie(resp, sid)
    return resp

@app.route('/api/clean', methods=['POST'])
def api_clean():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    raw_path = os.path.join(d, 'raw.csv')
    cleaned_path = os.path.join(d, 'cleaned.csv')
    if not os.path.exists(raw_path):
        return jsonify({'error': 'No raw file uploaded'}), 400
    try:
        payload = request.get_json(silent=True) or {}
        force = bool(payload.get('force', False))
        # If cleaned already exists and not forced, return it without re-running cleaning to avoid unnecessary alerts
        if os.path.exists(cleaned_path) and not force:
            try:
                existing = pd.read_csv(cleaned_path)
                preview = existing.head(1000).to_dict(orient='records')
                return jsonify({'columns': list(existing.columns), 'preview': preview, 'already_cleaned': True})
            except Exception:
                # fallthrough to re-clean if reading failed
                pass

        df = pd.read_csv(raw_path)
        cleaned = auto_clean(df)
        cleaned.to_csv(cleaned_path, index=False)
        preview = cleaned.head(1000).to_dict(orient='records')
        return jsonify({'columns': list(cleaned.columns), 'preview': preview, 'already_cleaned': False})
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


@app.route('/api/get-predictions', methods=['GET'])
def api_get_predictions():
    sid = make_session_id()
    d = ensure_session_dir(sid)
    preds_path = os.path.join(d, 'predictions.csv')
    if not os.path.exists(preds_path):
        return jsonify({'error': 'No predictions available'}), 400
    try:
        df = pd.read_csv(preds_path)
        preview = df.head(2000).to_dict(orient='records')
        return jsonify({'columns': list(df.columns), 'preview': preview, 'rows': len(df)})
    except Exception as e:
        return jsonify({'error':'Failed to read predictions','detail':str(e)}),500

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
    # Load cleaned if possible else raw; attempt basic cleaning if needed
    df = None
    if os.path.exists(cleaned_path):
        try:
            df = pd.read_csv(cleaned_path)
        except Exception:
            df = None
    if df is None and os.path.exists(raw_path):
        try:
            df = pd.read_csv(raw_path)
        except Exception:
            return jsonify({'error':'Failed to read raw data'}),400
    if df is None:
        return jsonify({'error':'No data available'}),400

    payload = request.json or {}
    time_col = payload.get('time_column')
    target = payload.get('target')
    model_name = payload.get('model','random_forest')
    horizon = int(payload.get('horizon',6) or 6)
    freq_override = payload.get('freq')  # 'D','W','M','Y' or None or 'auto'
    compare_flag = bool(payload.get('compare', False))
    plot_type = payload.get('plot_type','plotly')
    ml_model = payload.get('ml_model','random_forest')

    # Basic auto-clean if cleaned.csv absent or invalid: drop all-empty rows and strip header whitespace
    try:
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        df = df.dropna(how='all')
    except Exception:
        pass

    # Infer time_col if not provided
    if not time_col or time_col not in df.columns:
        candidates = [c for c in df.columns if c.lower() in ('date','time','datetime','timestamp') or 'date' in c.lower()]
        if candidates:
            time_col = candidates[0]
        else:
            for c in df.columns:
                try:
                    tmp = pd.to_datetime(df[c], errors='coerce')
                    if tmp.notnull().sum() > 0:
                        time_col = c
                        break
                except Exception:
                    continue

    # Infer target if not provided
    if not target or target not in df.columns:
        numerics = df.select_dtypes(include=[float,int]).columns.tolist()
        numerics = [c for c in numerics if c != time_col]
        if numerics:
            target = numerics[0]
        else:
            for c in df.columns:
                if c == time_col: continue
                try:
                    if pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce').notnull().sum() > 0:
                        target = c
                        break
                except Exception:
                    continue
    if not target or target not in df.columns:
        return jsonify({'error':'Target column not provided and could not be inferred'}),400

    # Prepare df_model
    orig_time_samples = df[time_col].dropna().astype(str).head(50).tolist() if time_col in df.columns else []
    df_model = df.copy()
    if time_col in df_model.columns:
        df_model[time_col] = pd.to_datetime(df_model[time_col], errors='coerce')
    df_model[target] = pd.to_numeric(df_model[target].astype(str).str.replace(',',''), errors='coerce')
    if time_col in df_model.columns:
        df_model = df_model[[time_col, target]].dropna(subset=[time_col, target]).sort_values(time_col).reset_index(drop=True)
    else:
        df_model = df_model[[target]].dropna(subset=[target])

    # Optional aggregation/resampling before modeling
    aggregate = bool(payload.get('aggregate', False))
    if aggregate and time_col in df_model.columns:
        try:
            df_model.set_index(time_col, inplace=True)
            # map freq to pandas offset alias
            freq_map = {'D':'D','W':'W','M':'M','Y':'A'}
            use_freq = freq if freq in freq_map else 'M'
            resample_alias = freq_map.get(use_freq,'M')
            method = (payload.get('aggregate_method') or 'sum').lower()
            if method == 'sum':
                df_resampled = df_model[target].resample(resample_alias).sum()
            elif method == 'mean':
                df_resampled = df_model[target].resample(resample_alias).mean()
            elif method == 'last':
                df_resampled = df_model[target].resample(resample_alias).last()
            else:
                df_resampled = df_model[target].resample(resample_alias).sum()
            df_resampled = df_resampled.fillna(method='ffill').to_frame().reset_index()
            df_resampled.columns = [time_col, target]
            df_model = df_resampled.sort_values(time_col).reset_index(drop=True)
        except Exception:
            # if resampling fails, fall back to original df_model
            pass

    # Frequency inference
    def infer_freq_from_series(ser_dt):
        ser = ser_dt.dropna().sort_values()
        if len(ser) < 2:
            return None
        diffs = ser.diff().dropna().dt.days
        if diffs.empty:
            return None
        med = float(diffs.median())
        if abs(med - 7) <= 1:
            return 'W'
        if 25 <= med <= 35:
            return 'M'
        if 360 <= med <= 370:
            return 'Y'
        if med <= 1.5:
            return 'D'
        if med < 14:
            return 'W'
        return None

    inferred_freq = None
    if time_col in df_model.columns and not df_model[time_col].isnull().all():
        inferred_freq = infer_freq_from_series(pd.to_datetime(df_model[time_col]))
    freq = freq_override if freq_override and freq_override != 'auto' else (inferred_freq or 'M')

    # detect original date string format
    def detect_date_format(samples):
        import re
        for s in samples:
            if not isinstance(s, str): continue
            s = s.strip()
            if re.match(r'^\\d{4}-\\d{2}-\\d{2}$', s): return '%Y-%m-%d'
            if re.match(r'^\\d{2}/\\d{2}/\\d{4}$', s): return '%d/%m/%Y'
            if re.match(r'^\\d{4}/\\d{2}/\\d{2}$', s): return '%Y/%m/%d'
            if re.match(r'^[A-Za-z]{3,}\\s+\\d{4}$', s): return '%b %Y'
            if re.match(r'^\\d{4}-\\d{2}$', s): return '%Y-%m'
            if re.match(r'^\\d{4}$', s): return '%Y'
        return None

    orig_fmt = detect_date_format(orig_time_samples)
    m_map = {'D':7, 'W':52, 'M':12, 'Y':1}
    m = m_map.get(freq, 12)

    try:
        do_arima = (model_name.lower() == 'arima') or compare_flag
        arima_preds = None
        arima_metrics = {'mse': None, 'r2': None}

        if do_arima:
            try:
                import pmdarima as pm
            except Exception:
                if model_name.lower() == 'arima' and not compare_flag:
                    return jsonify({'error':'pmdarima not installed on server. Install pmdarima to use ARIMA.'}),500
                do_arima = False

            if do_arima:
                series = df_model[target].astype(float).dropna()
                if len(series) >= 6:
                    try:
                        arima_model = pm.auto_arima(series, seasonal=True, m=m, suppress_warnings=True, error_action='ignore')
                        forecast = arima_model.predict(n_periods=horizon)

                        # build future dates based on freq
                        last_time = pd.to_datetime(df_model[time_col].iloc[-1]) if time_col in df_model.columns else None
                        future_dates = []
                        if last_time is not None:
                            for i in range(1, horizon+1):
                                if freq == 'D':
                                    future_dates.append(last_time + pd.DateOffset(days=i))
                                elif freq == 'W':
                                    future_dates.append(last_time + pd.DateOffset(weeks=i))
                                elif freq == 'Y':
                                    future_dates.append(last_time + pd.DateOffset(years=i))
                                else:
                                    future_dates.append(last_time + pd.DateOffset(months=i))
                            if orig_fmt:
                                period_labels = [d.strftime(orig_fmt) for d in future_dates]
                            else:
                                if freq == 'D':
                                    period_labels = [d.strftime('%Y-%m-%d') for d in future_dates]
                                elif freq == 'Y':
                                    period_labels = [d.strftime('%Y') for d in future_dates]
                                else:
                                    period_labels = [d.strftime('%b %Y') for d in future_dates]
                        else:
                            period_labels = [str(i) for i in range(1, horizon+1)]

                        arima_preds = pd.DataFrame({f'{target}_predicted': list(map(float, forecast))})
                        arima_preds['period'] = period_labels

                        try:
                            insample = arima_model.predict_in_sample()
                            minlen = min(len(insample), len(series))
                            mse = float(((insample[-minlen:] - series.values[-minlen:])**2).mean())
                            ss_res = ((series.values[-minlen:] - insample[-minlen:])**2).sum()
                            ss_tot = ((series.values[-minlen:] - series.values[-minlen:].mean())**2).sum()
                            r2 = float(1 - ss_res/ss_tot) if ss_tot!=0 else None
                            arima_metrics = {'mse':mse, 'r2':r2}
                        except Exception:
                            pass
                    except Exception as e:
                        if model_name.lower() == 'arima' and not compare_flag:
                            return jsonify({'error':'ARIMA fitting failed','detail':str(e),'trace':traceback.format_exc()}),500
                        arima_preds = None

        # ML lag-based fallback / compare
        ml_preds = None
        ml_metrics = {'mse': None, 'r2': None}
        if compare_flag or (model_name.lower() in ('random_forest','xgboost') and not do_arima):
            # Build richer lag/time features to improve multi-step forecasts
            def make_lag_features(series, times=None, n_lags=6):
                """Return a DataFrame with y and lag_1..lag_n_lags plus time-based features aligned to the y row.
                times (optional) should be same-length index-aligned pd.Series of datetimes for generating time features."""
                s = pd.Series(series).reset_index(drop=True)
                N = len(s)
                df_l = pd.DataFrame({'y': s})
                for i in range(1, n_lags+1):
                    df_l[f'lag_{i}'] = df_l['y'].shift(i)
                # drop initial rows with NaNs from lagging
                df_l = df_l.dropna().reset_index(drop=True)
                if times is not None and len(times) == N:
                    t = pd.to_datetime(times).reset_index(drop=True)
                    t = t.iloc[n_lags:].reset_index(drop=True)
                    df_l['time_ord'] = t.map(lambda d: d.toordinal() if not pd.isna(d) else 0)
                    df_l['month'] = t.dt.month.fillna(0).astype(int)
                else:
                    df_l['time_ord'] = list(range(len(df_l)))
                    df_l['month'] = 0
                # add rolling stats of recent values
                df_l['roll_mean_3'] = df_l[[f'lag_{i}' for i in range(1, min(4, n_lags+1))]].mean(axis=1)
                df_l['roll_std_3'] = df_l[[f'lag_{i}' for i in range(1, min(4, n_lags+1))]].std(axis=1).fillna(0)
                return df_l

            n_lags = int(payload.get('n_lags', 6) or 6)
            series = df_model[target].reset_index(drop=True)
            times = pd.to_datetime(df_model[time_col]).reset_index(drop=True) if time_col in df_model.columns else None
            lagdf = make_lag_features(series, times=times, n_lags=n_lags)
            if not lagdf.empty:
                # training features and target
                feature_cols = [c for c in lagdf.columns if c.startswith('lag_')] + ['time_ord', 'month', 'roll_mean_3', 'roll_std_3']
                X = lagdf[feature_cols].astype(float).fillna(0)
                y = lagdf['y'].astype(float).fillna(0)
                use_xgb = (ml_model.lower() == 'xgboost' and XGBOOST_AVAILABLE)
                if use_xgb:
                    model = XGBRegressor(objective='reg:squarederror', n_estimators=200)
                else:
                    model = RandomForestRegressor(n_estimators=200)
                model.fit(X, y)
                # prepare iterative forecasting using separate lag values and fresh time features each step
                full_last = X.values[-1].astype(float)
                # extract lag values slice (first n_lags columns)
                lag_vals = full_last[:n_lags].astype(float).copy()
                future = []
                future_idx = []
                last_time = pd.to_datetime(df_model[time_col].iloc[-1]) if time_col in df_model.columns else None
                for h in range(horizon):
                    # compute future time features
                    if last_time is not None:
                        if freq == 'D':
                            fut_time = last_time + pd.DateOffset(days=h+1)
                        elif freq == 'W':
                            fut_time = last_time + pd.DateOffset(weeks=h+1)
                        elif freq == 'Y':
                            fut_time = last_time + pd.DateOffset(years=h+1)
                        else:
                            fut_time = last_time + pd.DateOffset(months=h+1)
                        time_ord = float(fut_time.toordinal())
                        month = float(fut_time.month)
                    else:
                        time_ord = float(len(X) + h)
                        month = 0.0

                    # compute rolling stats from current lag_vals
                    try:
                        recent_lags_for_stats = lag_vals[:min(3, len(lag_vals))]
                        roll_mean_3 = float(np.mean(recent_lags_for_stats))
                        roll_std_3 = float(np.std(recent_lags_for_stats))
                    except Exception:
                        roll_mean_3 = 0.0
                        roll_std_3 = 0.0

                    # assemble feature vector: [lag_1..lag_n, time_ord, month, roll_mean_3, roll_std_3]
                    x_in = np.concatenate([lag_vals, np.array([time_ord, month, roll_mean_3, roll_std_3], dtype=float)])
                    # predict
                    try:
                        p = float(model.predict(x_in.reshape(1, -1))[0])
                    except Exception:
                        p = float(np.nan)
                    future.append(p)
                    # label for display
                    if last_time is not None:
                        if freq == 'D':
                            future_idx.append(fut_time.strftime('%Y-%m-%d'))
                        elif freq == 'W':
                            future_idx.append(fut_time.strftime('%Y-%m-%d'))
                        elif freq == 'Y':
                            future_idx.append(fut_time.strftime('%Y'))
                        else:
                            future_idx.append(fut_time.strftime('%b %Y'))
                    else:
                        future_idx.append(str(h+1))

                    # roll the lag values and insert prediction at lag_1
                    if len(lag_vals) > 1:
                        lag_vals = np.roll(lag_vals, 1)
                        lag_vals[0] = p
                    else:
                        lag_vals[0] = p
                ml_preds = pd.DataFrame({f'{target}_predicted': future})
                # If tree-based models (RandomForest/XGBoost) predict nearly constant values
                # over the horizon (a common issue when using time ordinal with trees),
                # fallback to a simple linear extrapolation using recent data trend.
                try:
                    vals = np.array(ml_preds[f'{target}_predicted'].astype(float).tolist())
                    if np.nanstd(vals) < 1e-6 or np.allclose(vals, vals[0], atol=1e-3):
                        # build linear trend from last observed points
                        k = min(8, len(series))
                        if k >= 2:
                            y_hist = np.array(series.dropna().astype(float).tolist()[-k:])
                            x_hist = np.arange(len(y_hist)).astype(float)
                            # fit linear trend (degree 1)
                            coef = np.polyfit(x_hist, y_hist, 1)
                            slope, intercept = float(coef[0]), float(coef[1])
                            last_idx = x_hist[-1]
                            linear_future = [intercept + slope * (last_idx + i + 1) for i in range(horizon)]
                            ml_preds[f'{target}_predicted'] = linear_future
                except Exception:
                    pass
                # Ensure there's always a human-friendly period label column
                if future_idx:
                    ml_preds['period'] = future_idx
                else:
                    # fallback to simple integer periods or based on last_time if available
                    if last_time is not None:
                        labels = []
                        for h in range(1, horizon+1):
                            if freq == 'D':
                                labels.append((last_time + pd.DateOffset(days=h)).strftime('%Y-%m-%d'))
                            elif freq == 'W':
                                labels.append((last_time + pd.DateOffset(weeks=h)).strftime('%Y-%m-%d'))
                            elif freq == 'Y':
                                labels.append((last_time + pd.DateOffset(years=h)).strftime('%Y'))
                            else:
                                labels.append((last_time + pd.DateOffset(months=h)).strftime('%b %Y'))
                        ml_preds['period'] = labels
                    else:
                        ml_preds['period'] = [str(i) for i in range(1, len(future)+1)]
                try:
                    from sklearn.metrics import mean_squared_error, r2_score
                    preds_train = model.predict(X)
                    mse_ml = float(mean_squared_error(y, preds_train))
                    r2_ml = float(r2_score(y, preds_train))
                    ml_metrics = {'mse':mse_ml,'r2':r2_ml}
                except Exception:
                    pass

        # Prepare outputs
        if model_name.lower() == 'arima' and not compare_flag:
            preds_df = arima_preds if arima_preds is not None else pd.DataFrame()
            preds_path = os.path.join(d, 'predictions.csv')
            if not preds_df.empty:
                preds_df.to_csv(preds_path, index=False)
                excel_bytes = df_to_excel_bytes(preds_df)
                if excel_bytes is not None:
                    with open(os.path.join(d,'predictions.xlsx'),'wb') as f:
                        f.write(excel_bytes.getvalue())
            if plot_type == 'matplotlib' and not preds_df.empty:
                try:
                    img = make_plot_image(history_df=df_model[[time_col,target]] if time_col in df_model.columns else None, preds_df=preds_df, title='ARIMA Forecast')
                    return jsonify({'metrics': arima_metrics, 'predictions_preview': preds_df.head(500).to_dict(orient='records'), 'plot_image': img})
                except Exception:
                    pass
            return jsonify({'metrics': arima_metrics, 'predictions_preview': preds_df.head(500).to_dict(orient='records')})

        if compare_flag:
            out = {'metrics': {'arima': arima_metrics, 'ml': ml_metrics}}
            previews = []
            if arima_preds is not None:
                previews.append({'model':'arima','preview': arima_preds.head(500).to_dict(orient='records')})
            if ml_preds is not None:
                previews.append({'model':'ml','preview': ml_preds.head(500).to_dict(orient='records')})
            out['predictions_preview'] = previews
            primary = arima_preds if arima_preds is not None else ml_preds
            if primary is not None and not primary.empty:
                primary.to_csv(os.path.join(d,'predictions.csv'), index=False)
                excel_bytes = df_to_excel_bytes(primary)
                if excel_bytes is not None:
                    with open(os.path.join(d,'predictions.xlsx'),'wb') as f:
                        f.write(excel_bytes.getvalue())
            if plot_type == 'matplotlib':
                try:
                    if arima_preds is not None:
                        out['plot_image_arima'] = make_plot_image(history_df=df_model[[time_col,target]] if time_col in df_model.columns else None, preds_df=arima_preds, title='ARIMA Forecast')
                    if ml_preds is not None:
                        out['plot_image_ml'] = make_plot_image(history_df=df_model[[time_col,target]] if time_col in df_model.columns else None, preds_df=ml_preds, title='ML Forecast')
                except Exception:
                    pass
            return jsonify(out)

        if ml_preds is not None and not compare_flag:
            preds_df = ml_preds
            preds_path = os.path.join(d, 'predictions.csv')
            preds_df.to_csv(preds_path, index=False)
            excel_bytes = df_to_excel_bytes(preds_df)
            if excel_bytes is not None:
                with open(os.path.join(d,'predictions.xlsx'),'wb') as f:
                    f.write(excel_bytes.getvalue())
            if plot_type == 'matplotlib':
                try:
                    img = make_plot_image(history_df=df_model[[time_col,target]] if time_col in df_model.columns else None, preds_df=preds_df, title='ML Forecast')
                    return jsonify({'metrics': ml_metrics, 'predictions_preview': preds_df.head(500).to_dict(orient='records'), 'plot_image': img})
                except Exception:
                    pass
            return jsonify({'metrics': ml_metrics, 'predictions_preview': preds_df.head(500).to_dict(orient='records')})

        return jsonify({'error':'No predictions could be generated (insufficient data or model failure)'}),400

    except Exception as e:
        return jsonify({'error':'Prediction failed','detail':str(e),'trace':traceback.format_exc()}),500

@app.route('/api/new-session', methods=['POST'])
def api_new_session():
    # create a new session id and set cookie
    sid = str(uuid.uuid4())
    ensure_session_dir(sid)
    resp = jsonify({'sid': sid})
    resp = set_session_cookie(resp, sid)
    return resp

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
