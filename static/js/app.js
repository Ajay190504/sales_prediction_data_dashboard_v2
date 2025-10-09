// app.js
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const btnClean = document.getElementById('btn-clean');
const colX = document.getElementById('col-x');
const colY = document.getElementById('col-y');
const colZ = document.getElementById('col-z');
const groupby = document.getElementById('groupby');
const timeCol = document.getElementById('time-col');
const targetCol = document.getElementById('target-col');
const chartArea = document.getElementById('chart-area');
const dataPreview = document.getElementById('data-preview');
const predPreview = document.getElementById('pred-preview');
const predMetrics = document.getElementById('pred-metrics');
const downloadCleanBtn = document.getElementById('download-clean');
const downloadPredsBtn = document.getElementById('download-preds');
const downloadChartBtn = document.getElementById('download-chart');

async function getSessionStatus() {
  const res = await fetch('/api/session-status');
  return await res.json();
}

async function loadSession() {
  // clear stale UI elements first
  dataPreview.innerHTML = '';
  document.getElementById('chart-area').innerHTML = '';
  document.getElementById('pred-preview').innerHTML = '';
  predMetrics.innerHTML = '';
  document.getElementById('download-clean').style.display='none';
  document.getElementById('download-preds').style.display='none';

  const s = await getSessionStatus();
  if (s && s.sid) {
    document.getElementById('session-badge').innerText = 'Session: ' + s.sid.slice(0,8);
  }
  if (s.has_cleaned) {
    document.getElementById('download-clean').style.display = 'inline-block';
    document.getElementById('download-clean').onclick = () => { window.location='/download/cleaned'; };
    await loadCleaned();
  }
  if (s.has_predictions) {
    document.getElementById('download-preds').style.display = 'inline-block';
    document.getElementById('download-preds').onclick = () => { window.location='/download/predictions'; };
    try {
      const rp = await fetch('/api/get-predictions');
      const jp = await rp.json();
  if (!jp.error) { renderPredsTable(jp.preview); renderPredsChart(jp.preview); }
    } catch (e) { /* ignore */ }
  }
  fillColumnSelectors(s.columns || []);
}

async function loadCleaned(){
  const res = await fetch('/api/get-cleaned');
  const j = await res.json();
  if (j.error) return;
  renderPreview(j.preview);
  document.getElementById('rows-info').innerText = `Rows: ${j.rows}`;
}

function fillColumnSelectors(cols){
  [colX, colY, colZ, groupby, timeCol, targetCol].forEach(sel => {
    sel.innerHTML = '<option value="">-- select --</option>';
    cols.forEach(c => { const opt = document.createElement('option'); opt.value=c; opt.text=c; sel.appendChild(opt); });
  });
  colY.size = Math.min(8, cols.length+1);
}

function renderPreview(rows){
  if (!rows || rows.length===0){ dataPreview.innerHTML='<p>No data</p>'; return; }
  const cols = Object.keys(rows[0]);
  let html = '<table class="table table-sm table-striped"><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>';
    cols.forEach(c => html += `<td>${r[c]!==null? r[c] : ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  dataPreview.innerHTML = html;
}

// Render predictions table into #pred-preview
function renderPredsTable(rows) {
  const container = document.getElementById('pred-preview');
  if (!rows || rows.length===0) { container.innerHTML = '<div class="text-muted">No predictions to show.</div>'; return; }
  const cols = Object.keys(rows[0]);
  let html = '<table class="table table-sm table-bordered"><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(r=>{ html += '<tr>'; cols.forEach(c=> html += `<td>${r[c]!==null? r[c] : ''}</td>`); html += '</tr>'; });
  html += '</tbody></table>';
  container.innerHTML = html;
}

// Render predictions chart (Plotly) based on user-selected pred-chart-type
function renderPredsChart(rows) {
  const chartTypeEl = document.getElementById('pred-chart-type');
  const chartType = chartTypeEl ? chartTypeEl.value : 'line';
  const container = document.getElementById('chart-area');
  if (!rows || rows.length === 0) {
    container.innerHTML = '<div class="text-muted">No predictions to plot.</div>';
    return;
  }
  const x = rows.map(r => r.period || Object.keys(r)[0]);
  const yname = Object.keys(rows[0]).find(k=>k.includes('_predicted')) || Object.keys(rows[0])[1];
  const y = rows.map(r => Number(r[yname]));
  const plotType = (chartType === 'area') ? 'scatter' : chartType;
  const trace = { x: x, y: y, type: plotType, mode: 'lines+markers' };
  if (chartType === 'area') trace.fill = 'tozeroy';
  Plotly.newPlot('chart-area', [trace], { title: 'Predicted future' });
}

fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  // client-side validation for extension
  const allowed = ['.csv','.xls','.xlsx'];
  const fname = (f.name || '').toLowerCase();
  const ext = fname.includes('.') ? fname.slice(fname.lastIndexOf('.')) : '';
  if (!allowed.includes(ext)) { toast('Upload','Unsupported file type. Use .csv, .xls or .xlsx'); fileInput.value = ''; return; }
  uploadStatus.innerText = 'Uploading...';
  const form = new FormData();
  form.append('file', f);
  try {
    const res = await fetch('/api/upload', { method:'POST', body: form });
    // handle non-OK status explicitly so we don't fall into the catch with ambiguous message
    if (!res.ok) {
      const txt = await res.text();
      let errObj = null;
      try { errObj = JSON.parse(txt); } catch(e) { errObj = null; }
      uploadStatus.innerText = `Upload error (${res.status})`;
      alert(JSON.stringify(errObj || { status: res.status, text: txt }));
      return;
    }
    let j = null;
    try { j = await res.json(); } catch (e) {
      // server returned non-JSON (or empty) — the file may still be saved.
      console.warn('Upload: failed to parse JSON response', e);
      // Refresh session status to detect that the raw file exists and expose columns if available
      try {
        const s = await getSessionStatus();
        if (s && (s.has_raw || s.has_cleaned)) {
          uploadStatus.innerText = 'Upload completed — raw file saved (preview unavailable).';
          // populate selectors if server provided columns
          if (s.columns && s.columns.length>0) fillColumnSelectors(s.columns);
          // if cleaned exists, load cleaned preview
          if (s.has_cleaned) await loadCleaned();
          btnClean.disabled = false;
          return;
        }
      } catch (ex) {
        console.error('Upload: unable to refresh session status after non-JSON upload response', ex);
      }
      uploadStatus.innerText = 'Upload completed (server returned unexpected response)';
      btnClean.disabled = false;
      return;
    }
    if (j.error) { uploadStatus.innerText = 'Upload error'; alert(JSON.stringify(j)); return; }
    uploadStatus.innerText = `Uploaded: ${f.name}`;
    fillColumnSelectors(j.columns || []);
    renderPreview(j.preview);
    btnClean.disabled = false;
  } catch (err) { uploadStatus.innerText='Upload failed'; console.error(err); }
});

btnClean.addEventListener('click', async () => {
  btnClean.innerText = 'Cleaning...';
  try {
    const force = document.getElementById('force-clean') && document.getElementById('force-clean').checked;
    const res = await fetch('/api/clean', { method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({force}) });
    const j = await res.json();
    if (j.error) { alert('Clean error: '+j.error); btnClean.innerText='Auto Clean'; return; }
    // Render cleaned preview in all cases (including already_cleaned)
    renderPreview(j.preview);
    fillColumnSelectors(j.columns || []);
    document.getElementById('download-clean').style.display='inline-block';
    document.getElementById('download-clean').onclick = () => { window.location='/download/cleaned'; };
    if (j.already_cleaned) {
      uploadStatus.innerText = 'Data already cleaned - loaded cleaned data.';
      toast('Clean', 'Data already cleaned - loaded cleaned data.');
    }
    btnClean.innerText = 'Auto Clean';
  } catch (err) { alert('Clean failed'); btnClean.innerText='Auto Clean'; console.error(err); }
});

document.getElementById('gen-chart').addEventListener('click', async () => {
  const kind = document.getElementById('chart-type').value;
  const x = colX.value;
  const y_selected = Array.from(colY.selectedOptions).map(o=>o.value);
  const y = y_selected.length === 0 ? null : (y_selected.length === 1 ? y_selected[0] : y_selected);
  const z = colZ.value;
  const gb = groupby.value;
  const payload = { kind, x, y, z, groupby: gb, title: '' };
  try {
    const res = await fetch('/api/visualize', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const j = await res.json();
    if (j.error) { alert('Viz error: '+j.error); return; }
    if (j.type==='pie') {
      const data=[{ type:'pie', labels:j.labels, values:j.values }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type==='heatmap') {
      const data=[{ z:j.z, x:j.x, y:j.y, type:'heatmap' }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type==='corr') {
      const z=j.matrix, x=j.cols, y=j.cols;
      const data=[{ z:z, x:x, y:y, type:'heatmap', colorscale:'RdBu' }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type==='multi') {
      const data = j.series.map(s=>({ x:s.x, y:s.y, name:s.name, mode:'lines+markers', type:'scatter' }));
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type==='line' || j.type==='bar' || j.type==='scatter' || j.type==='area') {
      const data=[{ x:j.x, y:j.y, type: j.type === 'area' ? 'scatter' : j.type, fill: j.type==='area' ? 'tozeroy' : undefined }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type==='histogram') {
      const data=[{ x:j.x, type:'histogram' }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type==='box') {
      const data=[{ y:j.y, type:'box' }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else {
      alert('Unknown plot response');
    }
    document.getElementById('download-chart').style.display='inline-block';
    document.getElementById('download-chart').onclick = async ()=>{
      try {
        const gd = document.getElementById('chart-area');
        const img = await Plotly.toImage(gd, {format:'png', width:1200, height:700});
        const a = document.createElement('a'); a.href = img; a.download = 'chart.png'; a.click();
      } catch (err) { alert('Chart download failed'); console.error(err); }
    };
  } catch (err) { console.error(err); alert('Chart generation failed'); }
});


document.getElementById('run-predict').addEventListener('click', async () => {
  const payload = {
    time_column: timeCol.value || null,
    target: targetCol.value || null,
    model: document.getElementById('model-select').value,
    ml_model: document.getElementById('model-ml-select') ? document.getElementById('model-ml-select').value : 'random_forest',
    freq: document.getElementById('freq-select') ? document.getElementById('freq-select').value : 'auto',
    plot_type: document.getElementById('plot-type') ? document.getElementById('plot-type').value : 'plotly',
    compare: document.getElementById('compare-models') ? document.getElementById('compare-models').checked : false,
    horizon: parseInt(document.getElementById('horizon').value||'6',10)
  };
  // include aggregate flag
  payload.aggregate = document.getElementById('aggregate-before') ? document.getElementById('aggregate-before').checked : false;
  payload.aggregate_method = document.getElementById('aggregate-method') ? document.getElementById('aggregate-method').value : 'sum';
  try {
    const res = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const j = await res.json();
    if (j.error) { alert('Predict error: '+(j.error||j.detail)); console.error(j); return; }

    if (j.metrics) {
      if (j.metrics.arima || j.metrics.ml) {
        predMetrics.innerHTML = `<div class="alert alert-info">ARIMA MSE: ${j.metrics.arima?.mse||'NA'} | ARIMA R²: ${j.metrics.arima?.r2||'NA'}<br>ML MSE: ${j.metrics.ml?.mse||'NA'} | ML R²: ${j.metrics.ml?.r2||'NA'}</div>`;
      } else {
        predMetrics.innerHTML = `<div class="alert alert-info">MSE: ${j.metrics.mse||'NA'} | R²: ${j.metrics.r2||'NA'}</div>`;
      }
    }

    const plotType = payload.plot_type || 'plotly';
    document.getElementById('compare-charts').style.display = 'none';
    document.getElementById('combined-table').style.display = 'none';
    if (plotType === 'matplotlib' && j.plot_image) {
      document.getElementById('chart-area').innerHTML = `<img src="data:image/png;base64,${j.plot_image}" style="max-width:100%"/>`;
    } else if (plotType === 'matplotlib' && (j.plot_image_arima || j.plot_image_ml)) {
      document.getElementById('chart-area').innerHTML = '';
      document.getElementById('compare-charts').style.display = 'flex';
      if (document.getElementById('chart-arima')) document.getElementById('chart-arima').innerHTML = j.plot_image_arima ? `<img src="data:image/png;base64,${j.plot_image_arima}" style="max-width:100%"/>` : '<div class="text-muted">No ARIMA image</div>';
      if (document.getElementById('chart-ml')) document.getElementById('chart-ml').innerHTML = j.plot_image_ml ? `<img src="data:image/png;base64,${j.plot_image_ml}" style="max-width:100%"/>` : '<div class="text-muted">No ML image</div>';
    } else {
      if (Array.isArray(j.predictions_preview) && j.predictions_preview.length>0 && j.predictions_preview[0].preview) {
        document.getElementById('chart-area').innerHTML = '';
        document.getElementById('compare-charts').style.display = 'flex';
        let arima = j.predictions_preview.find(p=>p.model==='arima');
        let ml = j.predictions_preview.find(p=>p.model==='ml');
        if (arima) {
          const x1 = arima.preview.map(r=>r.period);
          const y1name = Object.keys(arima.preview[0]).find(k=>k.includes('_predicted')) || Object.keys(arima.preview[0])[1];
          const y1 = arima.preview.map(r=>r[y1name]);
          Plotly.newPlot('chart-arima', [{x:x1,y:y1,type:'scatter',mode:'lines+markers'}], {title:'ARIMA Forecast'});
        } else document.getElementById('chart-arima').innerHTML = '<div class="text-muted">No ARIMA preview</div>';
        if (ml) {
          const x2 = ml.preview.map(r=>r.period);
          const y2name = Object.keys(ml.preview[0]).find(k=>k.includes('_predicted')) || Object.keys(ml.preview[0])[1];
          const y2 = ml.preview.map(r=>r[y2name]);
          Plotly.newPlot('chart-ml', [{x:x2,y:y2,type:'scatter',mode:'lines+markers'}], {title:'ML Forecast'});
        } else document.getElementById('chart-ml').innerHTML = '<div class="text-muted">No ML preview</div>';
        let tableHtml = '<table class="table table-sm table-bordered"><thead><tr><th>Period</th><th>ARIMA</th><th>ML</th></tr></thead><tbody>';
        const maxlen = Math.max(arima?.preview?.length||0, ml?.preview?.length||0);
        for (let i=0;i<maxlen;i++){
          const p = (arima && arima.preview[i]) ? arima.preview[i].period : (ml && ml.preview[i] ? ml.preview[i].period : '');
          const a = (arima && arima.preview[i]) ? Object.values(arima.preview[i]).find(v=>typeof v === 'number' || !isNaN(v)) : '';
          const m = (ml && ml.preview[i]) ? Object.values(ml.preview[i]).find(v=>typeof v === 'number' || !isNaN(v)) : '';
          tableHtml += `<tr><td>${p||''}</td><td>${a||''}</td><td>${m||''}</td></tr>`;
        }
        tableHtml += '</tbody></table>';
        document.getElementById('combined-table').style.display = 'block';
        document.getElementById('combined-table').innerHTML = tableHtml;
      } else if (j.predictions_preview && j.predictions_preview.length>0) {
        const x = j.predictions_preview.map(r=>r.period);
        const yname = Object.keys(j.predictions_preview[0]).find(k=>k.includes('_predicted')) || Object.keys(j.predictions_preview[0])[1];
        const y = j.predictions_preview.map(r=>r[yname]);
        Plotly.newPlot('chart-area', [{x:x,y:y,type:'scatter',mode:'lines+markers'}], {title:'Predicted future'});
      } else {
        document.getElementById('chart-area').innerHTML = '<div class="text-muted">No prediction preview returned.</div>';
      }
    }

    // (predictions table rendering is handled by top-level renderPredsTable)

    // handle different shapes
    if (j.predictions_preview) {
      if (Array.isArray(j.predictions_preview)) {
        // plain list of rows
        if (j.predictions_preview.length>0 && j.predictions_preview[0].preview) {
          // compare mode handled above; show combined table is already set
        } else {
          renderPredsTable(j.predictions_preview);
        }
      } else if (typeof j.predictions_preview === 'object') {
        // could be single DF-like object
        renderPredsTable(j.predictions_preview);
      }
    }

    if (document.getElementById('download-preds')) {
      document.getElementById('download-preds').style.display = 'inline-block';
      document.getElementById('download-preds').onclick = ()=>{ window.location='/download/predictions'; };
    }

    // if server didn't return a preview payload, try fetching stored predictions
    if (!j.predictions_preview) {
      try {
        const rp = await fetch('/api/get-predictions');
        const jp = await rp.json();
          if (!jp.error) { renderPredsTable(jp.preview); renderPredsChart(jp.preview); }
      } catch (e) { /* ignore */ }
    }

  } catch (err) { console.error(err); alert('Prediction failed: '+err.message); }
});

// Visualize the selected time series (time vs target)
document.getElementById('preview-series')?.addEventListener('click', async () => {
  const time = document.getElementById('time-col').value || null;
  const target = document.getElementById('target-col').value || null;
  if (!target) { toast('Viz','Select a target column to visualize'); return; }
  // Use the existing visualize API to render a line chart of time vs target
  const payload = { kind: 'line', x: time, y: target, title: 'Series: ' + (target) };
  try {
    const res = await fetch('/api/visualize', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const j = await res.json();
    if (j.error) { toast('Viz error', j.error); return; }
    // reuse the existing chart rendering logic by triggering the same as gen-chart
    if (j.type === 'line' || j.type === 'bar' || j.type === 'scatter' || j.type === 'area') {
      const data=[{ x: j.x, y: j.y, type: j.type === 'area' ? 'scatter' : j.type, fill: j.type==='area' ? 'tozeroy' : undefined }];
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else if (j.type === 'multi') {
      const data = j.series.map(s=>({ x:s.x, y:s.y, name:s.name, mode:'lines+markers', type:'scatter' }));
      Plotly.newPlot('chart-area', data, j.layout || {});
    } else {
      // fallback
      document.getElementById('chart-area').innerHTML = '<div class="text-muted">Cannot visualize selected series</div>';
    }
  } catch (err) { console.error(err); toast('Viz','Failed to generate series chart'); }
});


document.getElementById('btn-get-status').addEventListener('click', loadSession);
window.addEventListener('load', loadSession);

// Visualize stored predictions (fetch and render)
document.getElementById('visualize-predictions')?.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/get-predictions');
    const j = await res.json();
    if (j.error) { toast('Predictions', j.error); return; }
    if (j.preview && j.preview.length>0) {
      renderPredsTable(j.preview);
      renderPredsChart(j.preview);
      document.getElementById('download-preds').style.display = 'inline-block';
      document.getElementById('download-preds').onclick = ()=>{ window.location='/download/predictions'; };
      toast('Predictions', 'Loaded stored predictions');
    } else {
      toast('Predictions', 'No predictions available to visualize');
    }
  } catch (err) { console.error(err); toast('Predictions','Failed to load predictions'); }
});

// make Reload Session create a new session (fresh SID cookie) then load session state
document.getElementById('btn-get-status').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/new-session', { method: 'POST' });
    const j = await res.json();
    // cookie will be set by server; then load session
    if (j && j.sid) document.getElementById('session-badge').innerText = 'Session: ' + j.sid.slice(0,8);
    toast('Session', 'New session started');
    await loadSession();
  } catch (err) { console.error('Failed to create new session', err); await loadSession(); }
});

function toast(title, message, timeout=4000) {
  const id = 't' + Date.now();
  const html = `<div id="${id}" class="toast align-items-center text-bg-primary border-0 show" role="alert" aria-live="assertive" aria-atomic="true">
    <div class="d-flex"><div class="toast-body"><strong>${title}:</strong> ${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div></div>`;
  const container = document.getElementById('toast-container');
  if (!container) return;
  container.insertAdjacentHTML('beforeend', html);
  setTimeout(()=>{ const el = document.getElementById(id); if (el) el.remove(); }, timeout);
}
