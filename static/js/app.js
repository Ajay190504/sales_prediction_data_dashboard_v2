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
  const s = await getSessionStatus();
  if (s.has_cleaned) {
    document.getElementById('download-clean').style.display = 'inline-block';
    document.getElementById('download-clean').onclick = () => { window.location='/download/cleaned'; };
    await loadCleaned();
  }
  if (s.has_predictions) {
    document.getElementById('download-preds').style.display = 'inline-block';
    document.getElementById('download-preds').onclick = () => { window.location='/download/predictions'; };
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

fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  uploadStatus.innerText = 'Uploading...';
  const form = new FormData();
  form.append('file', f);
  try {
    const res = await fetch('/api/upload', { method:'POST', body: form });
    const j = await res.json();
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
    const res = await fetch('/api/clean', { method:'POST' });
    const j = await res.json();
    if (j.error) { alert('Clean error: '+j.error); btnClean.innerText='Auto Clean'; return; }
    renderPreview(j.preview);
    fillColumnSelectors(j.columns || []);
    document.getElementById('download-clean').style.display='inline-block';
    document.getElementById('download-clean').onclick = () => { window.location='/download/cleaned'; };
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
    time_column: timeCol.value,
    target: targetCol.value,
    model: document.getElementById('model-select').value,
    horizon: parseInt(document.getElementById('horizon').value||'6',10)
  };
  try {
    const res = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const j = await res.json();
    if (j.error) { alert('Predict error: '+j.error); return; }
    predMetrics.innerHTML = `<div class="alert alert-info">MSE: ${j.metrics.mse.toFixed(3)} | R²: ${j.metrics.r2.toFixed(3)}</div>`;
    renderPred(j.predictions_preview);
    document.getElementById('download-preds').style.display='inline-block';
    document.getElementById('download-preds').onclick = ()=>{ window.location='/download/predictions'; };
    if (j.predictions_preview && j.predictions_preview.length>0 && j.predictions_preview[0].period) {
      const x = j.predictions_preview.map(r=>r.period);
      const y = j.predictions_preview.map(r=>r[Object.keys(r).find(k=>k.includes('_predicted'))]);
      Plotly.newPlot('chart-area', [{x:x,y:y,type:'bar'}], {title:'Predicted future (monthly)'});
    }
  } catch (err) { console.error(err); alert('Prediction failed'); }
});

document.getElementById('btn-get-status').addEventListener('click', loadSession);
window.addEventListener('load', loadSession);
