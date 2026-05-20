'use strict';

// ── Constants ─────────────────────────────────────────
const CIRC = 213.6; // 2π × 34 (ring radius)
const EXT  = { csv: ['.csv'], json: ['.json'], txt: ['.txt', '.tsv'] };

// ── State ─────────────────────────────────────────────
let state = {
  step:    1,
  session: null,
  type:    'csv',
  files:   [],
};

// ── DOM refs ──────────────────────────────────────────
const $ = id => document.getElementById(id);

const stepEls      = { 1: $('step1'), 2: $('step2'), 3: $('step3'), 4: $('step4') };
const stepperItems = document.querySelectorAll('.stepper-item');
const typeCards    = document.querySelectorAll('.type-card');

const nextBtn1      = $('nextBtn1');
const prevBtn2      = $('prevBtn2');
const processBtn    = $('processBtn');
const browseBtn     = $('browseBtn');
const fileInput     = $('fileInput');
const uploadArea    = $('uploadArea');
const fileListEl    = $('fileList');
const filePreview   = $('filePreview');
const fileTypeSpan  = $('selectedFileType');
const ringFill      = $('ringFill');
const ringPct       = $('ringPct');
const progressRing  = $('progressRing');
const progressText  = $('progressText');
const logStream     = $('statusMessages');
const downloadBtn   = $('downloadBtn');
const viewReportBtn = $('viewReportBtn');
const newAnalysisBtn= $('newAnalysisBtn');
const reportModal   = $('reportModal');
const reportContent = $('reportContent');
const modalClose    = $('modalClose');

// ── Bootstrap ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  bindEvents();
  setRing(0);
});

// ── Event binding ─────────────────────────────────────
function bindEvents() {
  // File type cards
  typeCards.forEach(c => c.addEventListener('click', () => pickType(c.dataset.type)));

  // Wizard navigation
  nextBtn1.addEventListener('click',      () => goTo(2));
  prevBtn2.addEventListener('click',      () => goTo(1));
  processBtn.addEventListener('click',    run);
  newAnalysisBtn.addEventListener('click', reset);

  // File input
  browseBtn.addEventListener('click',  () => fileInput.click());
  fileInput.addEventListener('change', e  => ingest(Array.from(e.target.files)));

  // Drag & drop
  uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', e => { e.preventDefault(); uploadArea.classList.remove('drag-over'); });
  uploadArea.addEventListener('drop',      e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    ingest(Array.from(e.dataTransfer.files));
  });

  // Keyboard: allow Enter/Space on dropzone
  uploadArea.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  // Modal
  modalClose.addEventListener('click', closeModal);
  reportModal.addEventListener('click', e => { if (e.target === reportModal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  // Result actions
  downloadBtn.addEventListener('click',   downloadResults);
  viewReportBtn.addEventListener('click', viewReport);
}

// ── Wizard navigation ─────────────────────────────────
function goTo(step) {
  stepEls[state.step].classList.remove('active');
  stepEls[step].classList.add('active');
  state.step = step;
  syncStepper();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function syncStepper() {
  stepperItems.forEach((item, i) => {
    const s = i + 1;
    item.classList.toggle('active',    s === state.step);
    item.classList.toggle('completed', s <  state.step);
  });
}

// ── File type selection ───────────────────────────────
function pickType(type) {
  state.type = type;
  typeCards.forEach(c => {
    const sel = c.dataset.type === type;
    c.classList.toggle('selected', sel);
    c.setAttribute('aria-checked', sel);
  });
  fileTypeSpan.textContent = type.toUpperCase();
  nextBtn1.disabled = false;
}

// ── File ingestion ────────────────────────────────────
function ingest(incoming) {
  const valid = incoming.filter(f => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    return EXT[state.type].includes(ext);
  });

  if (!valid.length) {
    log('warn', `No valid .${state.type} files found — try again`);
    return;
  }

  state.files = valid;
  renderFileList();
  processBtn.disabled = false;
}

function renderFileList() {
  fileListEl.innerHTML = '';

  state.files.forEach((f, i) => {
    const li = document.createElement('li');
    li.className = 'file-item';
    li.innerHTML = `
      <span class="file-item-icon" aria-hidden="true">
        <i class="fas fa-file-lines"></i>
      </span>
      <div class="file-item-info">
        <div class="file-item-name" title="${escHtml(f.name)}">${escHtml(f.name)}</div>
        <div class="file-item-size">${fmtSize(f.size)}</div>
      </div>
      <button class="file-item-remove" aria-label="Remove ${escHtml(f.name)}" data-index="${i}">
        <i class="fas fa-xmark" aria-hidden="true"></i>
      </button>`;
    fileListEl.appendChild(li);
  });

  // Delegate removal clicks
  fileListEl.querySelectorAll('.file-item-remove').forEach(btn => {
    btn.addEventListener('click', () => removeFile(+btn.dataset.index));
  });

  filePreview.hidden = state.files.length === 0;
}

function removeFile(i) {
  state.files.splice(i, 1);
  renderFileList();
  if (!state.files.length) processBtn.disabled = true;
}

function fmtSize(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / 1024 ** i).toFixed(1) + ' ' + units[i];
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Processing pipeline ───────────────────────────────
async function run() {
  if (!state.files.length) return;
  goTo(3);

  try {
    await upload();
    await process();
  } catch (err) {
    log('error', `Failed — ${err.message}`);
    console.error(err);
  }
}

async function upload() {
  const fd = new FormData();
  state.files.forEach(f => fd.append('files', f));
  fd.append('fileType', state.type);

  log('info', 'Uploading files…');
  setRing(10, 'Uploading…');

  const res = await fetch('/api/upload', { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`Upload failed — ${res.statusText}`);

  const data = await res.json();
  state.session = data.sessionId;

  log('success', `${data.files.length} file${data.files.length !== 1 ? 's' : ''} uploaded`);
  setRing(25, 'Upload complete');
}

async function process() {
  log('info', 'Starting analysis…');
  setRing(30, 'Analyzing…');

  const es = new EventSource(`/api/process/${state.session}`);

  es.onmessage = e => {
    const d = JSON.parse(e.data);
    setRing(d.progress, d.message);
    log('info', d.message);

    if (d.status === 'complete') {
      es.close();
      log('success', 'Analysis complete');
      setRing(100, 'Done');
      setTimeout(() => showResults(d.results), 800);
    } else if (d.status === 'error') {
      es.close();
      log('error', d.message);
    }
  };

  es.onerror = () => es.close();
}

// ── Progress ring ─────────────────────────────────────
function setRing(pct, text) {
  const pctRounded = Math.round(pct);
  const offset = CIRC * (1 - pct / 100);

  ringFill.style.strokeDashoffset = offset;
  ringPct.textContent = `${pctRounded}%`;
  progressRing.setAttribute('aria-valuenow', pctRounded);

  if (text) progressText.textContent = text;
}

// ── Log stream ────────────────────────────────────────
function log(type, msg) {
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-dot ${type}" aria-hidden="true"></span><span>${escHtml(msg)}</span>`;
  logStream.appendChild(line);
  logStream.scrollTop = logStream.scrollHeight;

  // Keep the log tidy
  while (logStream.children.length > 10) {
    logStream.removeChild(logStream.firstChild);
  }
}

// ── Results ───────────────────────────────────────────
function showResults(r) {
  const orig    = r.originalShape[0];
  const cleaned = r.cleanedShape[0];
  const quality = orig > 0 ? Math.round((cleaned / orig) * 100) : 100;

  $('filesProcessed').textContent = r.filesProcessed;
  $('issuesFound').textContent    = r.issuesFound;
  $('dataQuality').textContent    = `${quality}%`;
  $('rowsProcessed').textContent  = cleaned.toLocaleString();
  $('previewContent').innerHTML   = `<pre>${escHtml(r.reportPreview)}</pre>`;

  goTo(4);
}

// ── Download ──────────────────────────────────────────
async function downloadResults() {
  if (!state.session) return;

  try {
    const res = await fetch(`/api/download/${state.session}`);
    if (!res.ok) throw new Error(res.statusText);

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), {
      href: url,
      download: `dsaral_${state.session}.zip`,
    });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    log('error', `Download failed — ${err.message}`);
  }
}

// ── Report modal ──────────────────────────────────────
async function viewReport() {
  if (!state.session) return;

  try {
    const res = await fetch(`/api/report/${state.session}`);
    if (!res.ok) throw new Error(res.statusText);

    const data = await res.json();
    reportContent.textContent = data.report;
    reportModal.hidden = false;
    modalClose.focus();
  } catch (err) {
    log('error', `Could not load report — ${err.message}`);
  }
}

function closeModal() {
  reportModal.hidden = true;
}

// ── Reset ─────────────────────────────────────────────
function reset() {
  state = { step: 1, session: null, type: 'csv', files: [] };

  typeCards.forEach(c => { c.classList.remove('selected'); c.setAttribute('aria-checked', 'false'); });
  nextBtn1.disabled   = true;
  processBtn.disabled = true;
  fileInput.value     = '';
  filePreview.hidden  = true;
  fileListEl.innerHTML = '';
  logStream.innerHTML  = '';
  setRing(0, 'Initializing…');

  goTo(1);
}

// ── Cleanup on leave ──────────────────────────────────
window.addEventListener('beforeunload', () => {
  if (state.session) {
    fetch(`/api/cleanup/${state.session}`, { method: 'DELETE' }).catch(() => {});
  }
});