/* Signature Verification Console.
 *
 * Rule: every number and sentence describing a result comes from the backend
 * response (/api/v1/signature/inspect). The page only formats and lays it out.
 * All server text is inserted with textContent, never as HTML.
 */
(() => {
  'use strict';

  const API = {
    health: '/health',
    inspect: '/api/v1/signature/inspect',
    samples: '/api/v1/samples',
    sampleImage: (id) => `/api/v1/samples/${encodeURIComponent(String(id))}`,
  };
  const MAX_BYTES = 20 * 1024 * 1024;
  const ACCEPTED = /\.(png|jpe?g|tiff?|bmp)$/i;
  const HISTORY_LIMIT = 15;
  // Fixed colours baked into the server-rendered PNGs (src/api/visuals.py).
  const PNG_COLOURS = { reference: '#1F60A0', questioned: '#2F3437', keypoint: '#9F2F2D' };
  const BANDS = {
    ACCEPT: { label: 'Match', icon: 'M5 12.5l4.2 4.2L19 7' },
    REVIEW: { label: 'Manual review', icon: 'M12 6.5v7M12 17.5v.01' },
    REJECT: { label: 'No match', icon: 'M7 7l10 10M17 7L7 17' },
    INCONCLUSIVE: { label: 'Inconclusive', icon: 'M7 12h10' },
  };

  const state = {
    inputs: { reference: null, questioned: null },   // { dataUrl, name }
    sample: null,          // sample pair currently loaded in both zones, if any
    samples: [],
    history: [],
    activeHistoryId: null,
    busy: false,
    seq: 0,
  };

  const $ = (sel, root = document) => root.querySelector(sel);

  /** Create an element; children may be strings (as text) or nodes. */
  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'style') node.setAttribute('style', value);
      else node.setAttribute(key, value === true ? '' : String(value));
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function svgIcon(path) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    const p = document.createElementNS(ns, 'path');
    p.setAttribute('d', path);
    svg.append(p);
    return svg;
  }

  const fmt = (value, digits = 2) =>
    (value === null || value === undefined || Number.isNaN(value)) ? '–' : Number(value).toFixed(digits);
  const signed = (value, digits = 2) =>
    (value === null || value === undefined) ? '–' : (value >= 0 ? '+' : '−') + Math.abs(value).toFixed(digits);
  const pngSrc = (b64) => `data:image/png;base64,${b64}`;

  // ------------------------------------------------------------------ alerts
  function showAlert(message) {
    const box = $('#alert');
    box.textContent = message;
    box.hidden = false;
  }
  function clearAlert() { $('#alert').hidden = true; }

  // ------------------------------------------------------------------ health
  async function loadHealth() {
    const dot = $('#engine-dot');
    const text = $('#engine-text');
    try {
      const res = await fetch(API.health);
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      dot.className = 'dot ok';
      text.textContent = `Engine ${data.versions.model} · ${data.versions.policy}`;
    } catch (err) {
      dot.className = 'dot bad';
      text.textContent = 'Engine unreachable';
    }
  }

  // ------------------------------------------------------------------ inputs
  const zones = {};
  function setupZone(zone) {
    const role = zone.dataset.role;
    const input = $('input[type=file]', zone);
    zones[role] = zone;

    zone.addEventListener('click', (e) => {
      if (e.target.closest('.zone-clear')) return;
      input.click();
    });
    zone.addEventListener('keydown', (e) => {
      if (e.target !== zone) return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); clearZone(role); }
    });
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) acceptFile(role, input.files[0]);
      input.value = '';
    });
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) acceptFile(role, file);
    });
    $('.zone-clear', zone).addEventListener('click', (e) => { e.stopPropagation(); clearZone(role); });
  }

  function readAsDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error('read failed'));
      reader.readAsDataURL(blob);
    });
  }

  async function acceptFile(role, file, { fromSample = false } = {}) {
    clearAlert();
    if (file.size > MAX_BYTES) { showAlert('Image exceeds the 20 MB limit.'); return; }
    const typeOk = /^image\/(png|jpeg|tiff|bmp)$/.test(file.type) || ACCEPTED.test(file.name || '');
    if (!typeOk) { showAlert('Unsupported file type. Use PNG, JPEG, TIFF or BMP.'); return; }
    try {
      const dataUrl = await readAsDataUrl(file);
      setZone(role, dataUrl, file.name || 'pasted-image.png');
      if (!fromSample) state.sample = null;
    } catch (err) {
      showAlert('The file could not be read.');
    }
  }

  function setZone(role, dataUrl, name) {
    const zone = zones[role];
    state.inputs[role] = { dataUrl, name };
    const img = $('.zone-preview', zone);
    // TIFF previews are not supported by most browsers; the file is still sent as-is.
    img.src = dataUrl;
    img.alt = `${role === 'reference' ? 'Reference specimen' : 'Signature to verify'} preview`;
    img.hidden = false;
    img.onerror = () => { img.hidden = true; };
    $('.zone-empty', zone).hidden = true;
    $('.zone-file', zone).textContent = name;
    $('.zone-clear', zone).hidden = false;
    zone.classList.add('filled');
    updateVerifyButton();
  }

  function clearZone(role) {
    const zone = zones[role];
    state.inputs[role] = null;
    state.sample = null;
    const img = $('.zone-preview', zone);
    img.hidden = true;
    img.removeAttribute('src');
    $('.zone-empty', zone).hidden = false;
    $('.zone-file', zone).textContent = 'No file';
    $('.zone-clear', zone).hidden = true;
    zone.classList.remove('filled');
    updateVerifyButton();
  }

  function updateVerifyButton() {
    $('#verify').disabled = state.busy || !(state.inputs.reference && state.inputs.questioned);
  }

  function setupPaste() {
    document.addEventListener('paste', (e) => {
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const item of items) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (!file) continue;
          e.preventDefault();
          const focused = document.activeElement && document.activeElement.closest && document.activeElement.closest('.zone');
          const role = focused ? focused.dataset.role
            : (!state.inputs.reference ? 'reference' : 'questioned');
          acceptFile(role, file);
          return;
        }
      }
    });
    // A file dropped outside a zone must not navigate away from the console.
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());
  }

  // ------------------------------------------------------------------ samples
  async function loadSamples() {
    try {
      const res = await fetch(API.samples);
      if (!res.ok) return;   // 404: gallery disabled on this server
      const data = await res.json();
      state.samples = Array.isArray(data.samples) ? data.samples : [];
    } catch (err) {
      return;
    }
    if (!state.samples.length) return;
    const select = $('#sample-select');
    select.append(el('option', { value: '', text: `Choose one of ${state.samples.length} labelled pairs…` }));
    for (const s of state.samples) {
      const writer = s.writer ? ` · ${s.writer}` : '';
      select.append(el('option', { value: String(s.id), text: `#${s.id} · ${s.dataset_label}${writer} · ${s.reference_name} vs ${s.questioned_name}` }));
    }
    select.addEventListener('change', () => {
      const pair = state.samples.find((s) => String(s.id) === select.value);
      if (pair) loadSamplePair(pair, false);
    });
    $('#sample-next').addEventListener('click', nextSample);
    $('#samples').hidden = false;
  }

  async function fetchSampleFile(id, name) {
    const res = await fetch(API.sampleImage(id));
    if (!res.ok) throw new Error(`sample ${id}: HTTP ${res.status}`);
    const blob = await res.blob();
    return new File([blob], name, { type: blob.type || 'image/png' });
  }

  async function loadSamplePair(pair, autoVerify) {
    clearAlert();
    try {
      const [ref, que] = await Promise.all([
        fetchSampleFile(pair.reference_image_id, pair.reference_name),
        fetchSampleFile(pair.questioned_image_id, pair.questioned_name),
      ]);
      await acceptFile('reference', ref, { fromSample: true });
      await acceptFile('questioned', que, { fromSample: true });
      state.sample = pair;
      $('#sample-select').value = String(pair.id);
      if (autoVerify) verify();
    } catch (err) {
      showAlert('The sample pair could not be loaded.');
    }
  }

  function nextSample() {
    if (!state.samples.length || state.busy) return;
    const current = state.samples.findIndex((s) => state.sample && s.id === state.sample.id);
    const pair = state.samples[(current + 1) % state.samples.length];
    loadSamplePair(pair, true);
  }

  // ------------------------------------------------------------------ verify
  async function verify() {
    if (state.busy || !(state.inputs.reference && state.inputs.questioned)) return;
    clearAlert();
    state.busy = true;
    const button = $('#verify');
    button.classList.add('busy');
    $('.btn-label', button).textContent = 'Verifying…';
    updateVerifyButton();
    const inputs = { reference: state.inputs.reference, questioned: state.inputs.questioned };
    const sample = state.sample;
    try {
      const res = await fetch(API.inspect, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference_images: [inputs.reference.dataUrl], questioned_image: inputs.questioned.dataUrl }),
      });
      let data = null;
      try { data = await res.json(); } catch (err) { data = null; }
      if (!res.ok) {
        const detail = data && typeof data.detail === 'string' ? data.detail : `Request failed (HTTP ${res.status}).`;
        showAlert(detail);
        return;
      }
      const entry = {
        id: ++state.seq,
        time: new Date(),
        data,
        sample,
        names: { reference: inputs.reference.name, questioned: inputs.questioned.name },
      };
      state.history.unshift(entry);
      state.history.length = Math.min(state.history.length, HISTORY_LIMIT);
      showEntry(entry);
    } catch (err) {
      showAlert('The verification service could not be reached.');
    } finally {
      state.busy = false;
      button.classList.remove('busy');
      $('.btn-label', button).textContent = 'Verify signature';
      updateVerifyButton();
    }
  }

  function showEntry(entry) {
    state.activeHistoryId = entry.id;
    renderResult(entry);
    renderPipeline(entry.data);
    renderHistory();
  }

  // ------------------------------------------------------------------ result header
  function renderResult(entry) {
    const { data, sample } = entry;
    const c = data.comparison;
    const band = BANDS[c.band] ? c.band : 'INCONCLUSIVE';
    const badge = $('#badge');
    badge.className = `badge ${band}`;
    const icon = $('#badge-icon');
    icon.replaceChildren(svgIcon(BANDS[band].icon));
    $('#badge-text').textContent = BANDS[band].label;
    $('#verdict-full').textContent = c.verdict;

    const margin = $('#margin');
    margin.replaceChildren();
    if (data.margin) {
      margin.append(el('span', { class: 'mono', text: data.margin.statement }));
    } else if (c.notes && c.notes.length) {
      margin.textContent = c.notes[0];
    }

    const ds = $('#dataset-label');
    if (sample) {
      ds.replaceChildren('Dataset label', el('strong', { text: sample.dataset_label }),
        el('span', { class: 'mono', text: `#${sample.id}` }));
      ds.hidden = false;
    } else {
      ds.hidden = true;
    }

    renderGauge(c.match_score, data.thresholds);

    const facts = [
      ['Log-odds', c.log_odds === null ? '–' : signed(c.log_odds, 3)],
      ['MATCH threshold', c.accept_threshold === null ? '–' : signed(c.accept_threshold, 3)],
      ['NO MATCH threshold', c.reject_threshold === null ? '–' : signed(c.reject_threshold, 3)],
      ['Consistent keypoints', c.consistent_stroke_features === null ? '–' : String(c.consistent_stroke_features)],
      ['Specimens used', `${c.references_used} / ${c.references_supplied}`],
      ['Processing time', `${fmt(data.timing.total_ms, 0)} ms`],
    ];
    $('#facts').replaceChildren(...facts.map(([k, v]) => el('div', {}, [el('dt', { text: k }), el('dd', { text: v })])));
    $('#result').hidden = false;
  }

  function renderGauge(score, t) {
    const lo = t.score_reject_anchor;
    const hi = t.score_accept_anchor;
    const zonesEls = document.querySelectorAll('.gauge-zone');
    zonesEls[0].style.width = `${lo}%`;
    zonesEls[1].style.width = `${hi - lo}%`;
    zonesEls[2].style.width = `${100 - hi}%`;
    const ticks = $('#gauge-ticks');
    ticks.replaceChildren(
      el('span', { class: 'gauge-tick edge-left', style: 'left:0%', text: '0' }),
      el('span', { class: 'gauge-tick', style: `left:${lo}%`, text: `NO MATCH ${fmt(lo, 0)}` }),
      el('span', { class: 'gauge-tick', style: `left:${hi}%`, text: `MATCH ${fmt(hi, 0)}` }),
      el('span', { class: 'gauge-tick edge-right', style: 'left:100%', text: '100' }),
    );
    const marker = $('#gauge-marker');
    if (score === null || score === undefined) {
      $('#score').textContent = '–';
      marker.hidden = true;
    } else {
      $('#score').textContent = fmt(score, 1);
      marker.hidden = false;
      marker.style.left = `${Math.max(0, Math.min(100, score))}%`;
    }
  }

  // ------------------------------------------------------------------ pipeline
  function figure(b64, caption, alt) {
    const frame = el('div', { class: 'figure-frame' },
      b64 ? el('img', { src: pngSrc(b64), alt, loading: 'lazy' }) : el('div', { class: 'figure-missing', text: 'Stage did not run for this image.' }));
    return el('figure', { class: 'figure' }, [frame, el('figcaption', {}, caption)]);
  }

  const kv = (key, value) => el('span', { class: 'kv' }, [el('span', { text: `${key} ` }), value]);

  function pairFigures(data, key, captionFor, altPrefix) {
    return el('div', { class: 'pair' }, ['reference', 'questioned'].map((role) => {
      const img = data[role];
      const title = role === 'reference' ? 'Reference' : 'Questioned';
      return figure(img[key], [el('strong', { text: title }), ...captionFor(img)], `${altPrefix}: ${title.toLowerCase()}`);
    }));
  }

  function stage(index, title, desc, content) {
    const node = $('#stage-template').content.firstElementChild.cloneNode(true);
    node.style.setProperty('--i', String(index));
    $('.stage-num', node).textContent = String(index + 1);
    $('.stage-title', node).textContent = title;
    $('.stage-desc', node).textContent = desc;
    $('.stage-content', node).append(...[].concat(content));
    return node;
  }

  const notRun = (text) => el('div', { class: 'note', text });

  function renderPipeline(data) {
    const c = data.comparison;
    const stages = [];
    const errors = ['reference', 'questioned']
      .filter((r) => data[r].error)
      .map((r) => el('div', { class: 'note warn', text: `${r === 'reference' ? 'Reference' : 'Questioned'}: ${data[r].error}` }));

    stages.push(stage(0, 'Original', 'The submitted images, downscaled for display only.',
      pairFigures(data, 'original_png', (img) => [kv('size', `${img.width} × ${img.height} px`)], 'Original')));

    stages.push(stage(1, 'Signature extraction',
      'Background flattened and ink level harmonised; the box marks the signature region the engine located.',
      [pairFigures(data, 'harmonised_png',
        (img) => (img.bbox ? [kv('box', `x ${img.bbox[0]}, y ${img.bbox[1]}, ${img.bbox[2]} × ${img.bbox[3]}`)] : []),
        'Harmonised image with signature box'), ...errors]));

    stages.push(stage(2, 'Normalisation', 'Tight crop of the ink-darkness map that every descriptor is computed on.',
      pairFigures(data, 'ink_crop_png',
        (img) => (img.stats ? [kv('aspect', fmt(img.stats.aspect_ratio, 3)), kv('ink density', fmt(img.stats.ink_density, 4))] : []),
        'Normalised ink crop')));

    stages.push(stage(3, 'Stroke representation',
      'Stroke centre-lines (skeleton) and the local keypoints centred on a stroke, on the shared comparison canvas. Coarse-scale keypoints centred off the strokes are counted but not drawn.',
      [pairFigures(data, 'strokes_png',
        (img) => (img.stats ? [kv('skeleton pts', String(img.stats.skeleton_points)), kv('keypoints', String(img.stats.keypoints)),
          kv('off-stroke, not drawn', String(img.stats.keypoints_off_stroke ?? 0)),
          kv('pen width', `${fmt(img.stats.stroke_width_px, 2)} px`)] : []),
        'Skeleton and keypoints'),
      el('div', { class: 'legend' }, [
        el('span', {}, [el('span', { class: 'swatch', style: `background:${PNG_COLOURS.questioned}` }), 'Skeleton']),
        el('span', {}, [el('span', { class: 'swatch', style: `background:${PNG_COLOURS.keypoint}` }), 'Keypoint']),
      ])]));

    stages.push(stage(4, 'Alignment & comparison',
      'Reference strokes laid over the questioned strokes on the comparison canvas.', renderAlignment(data)));

    stages.push(stage(5, 'Signal contributions',
      'Each fused signal is multiplied by its fitted weight; the sum plus the model bias is the log-odds.', renderFusion(data)));

    stages.push(stage(6, 'Similarity', 'Log-odds mapped onto the decision-aligned 0–100 scale.', renderSimilarity(data)));

    stages.push(stage(7, 'Decision', 'Verdict, operating-point notes and the engine\'s evidence report.', renderDecision(c)));

    $('#stages').replaceChildren(...stages);
    $('#pipeline').hidden = false;
  }

  function renderAlignment(data) {
    const a = data.alignment;
    if (!a) return notRun('Not computed: the pair was not scored (see Decision).');
    const facts = [
      kv('transform', a.transform_source === 'keypoint_ransac' ? 'keypoint RANSAC' : 'none (unaligned)'),
      a.rotation_deg !== null ? kv('rotation', `${fmt(a.rotation_deg, 2)}°`) : null,
      a.scale !== null ? kv('scale', fmt(a.scale, 3)) : null,
      kv('inliers / tentative', `${a.keypoint_inliers} / ${a.tentative_matches}`),
      kv('keypoints ref / que', `${a.reference_keypoints} / ${a.questioned_keypoints}`),
      kv('layout aligned', a.layout_alignment_used ? 'yes' : 'no'),
    ].filter(Boolean);
    const side = el('div', {}, [
      el('div', { class: 'note', text: a.description }),
      el('div', { class: 'legend' }, [
        el('span', {}, [el('span', { class: 'swatch', style: `background:${PNG_COLOURS.reference}` }), 'Reference (mapped)']),
        el('span', {}, [el('span', { class: 'swatch', style: `background:${PNG_COLOURS.questioned}` }), 'Questioned']),
      ]),
      el('div', { class: 'legend' }, facts),
    ]);
    return el('div', { class: 'overlay-wrap' }, [
      figure(a.overlay_png, [], 'Reference strokes overlaid on questioned strokes'),
      side,
    ]);
  }

  function renderFusion(data) {
    const f = data.fusion;
    if (!f) {
      if (data.comparison.band === 'INCONCLUSIVE') return notRun('Not computed: the pair was not scored (see Decision).');
      return el('div', { class: 'note warn', text: 'Signal breakdown withheld: the recomputed log-odds did not reproduce the verdict exactly.' });
    }
    const fused = f.signals.filter((s) => s.fused);
    const unfused = f.signals.filter((s) => !s.fused);
    const maxAbs = Math.max(1e-9, ...fused.map((s) => Math.abs(s.contribution)));
    const head = el('tr', {}, [
      el('th', { text: 'Signal' }), el('th', { class: 'num', text: 'Value' }), el('th', { class: 'num', text: 'Weight' }),
      el('th', { class: 'bar-cell', text: 'Contribution (weight × value)' }), el('th', { class: 'num', text: 'Log-odds' }),
    ]);
    const rows = fused.map((s) => el('tr', {}, [
      el('td', { text: s.label }),
      el('td', { class: 'num', text: fmt(s.value, 3) }),
      el('td', { class: 'num', text: fmt(s.weight, 3) }),
      el('td', { class: 'bar-cell' }, el('div', {
        class: `bar${s.contribution < 0 ? ' neg' : ''}`,
        style: `width:${(Math.abs(s.contribution) / maxAbs) * 100}%`,
        role: 'img', 'aria-label': `${s.label} contributes ${signed(s.contribution)}`,
      })),
      el('td', { class: 'num', text: signed(s.contribution) }),
    ]));
    rows.push(el('tr', { class: 'bias' }, [el('td', { text: 'Model bias' }), el('td'), el('td'), el('td'), el('td', { class: 'num', text: signed(f.bias) })]));
    rows.push(el('tr', { class: 'total' }, [el('td', { text: 'Log-odds' }), el('td'), el('td'), el('td'), el('td', { class: 'num', text: signed(f.log_odds, 3) })]));
    for (const s of unfused) {
      rows.push(el('tr', { class: 'unfused' }, [
        el('td', { text: `${s.label} (reported, not fused)` }), el('td', { class: 'num', text: fmt(s.value, 3) }),
        el('td', { class: 'num', text: '–' }), el('td'), el('td', { class: 'num', text: '–' }),
      ]));
    }
    const table = el('table', { class: 'contrib' }, [el('thead', {}, head), el('tbody', {}, rows)]);
    const equation = el('div', { class: 'equation' }, [
      'Σ contributions ', el('b', { text: signed(f.contributions_total, 3) }),
      '  +  bias ', el('b', { text: signed(f.bias, 3) }),
      '  =  log-odds ', el('b', { text: signed(f.log_odds, 3) }),
    ]);
    return [el('div', { style: 'overflow-x:auto' }, table), equation];
  }

  function renderSimilarity(data) {
    const c = data.comparison;
    const t = data.thresholds;
    if (c.match_score === null) return notRun('No score: the pair was not scored (see Decision).');
    return el('div', { class: 'equation' }, [
      'log-odds ', el('b', { text: signed(c.log_odds, 3) }), '  →  score ', el('b', { text: fmt(c.match_score, 1) }),
      `   (anchors: ${signed(t.reject_log_odds, 2)} → ${fmt(t.score_reject_anchor, 0)}, `,
      `${signed(t.accept_log_odds, 2)} → ${fmt(t.score_accept_anchor, 0)}, `,
      `±${fmt(t.score_tail_log_odds, 0)} beyond → 0 / 100)`,
    ]);
  }

  function renderDecision(c) {
    const parts = [el('div', { class: 'equation' }, ['verdict ', el('b', { text: c.verdict }), '   band ', el('b', { text: c.band })])];
    if (c.notes && c.notes.length) parts.push(el('ul', { class: 'notes' }, c.notes.map((n) => el('li', { text: n }))));
    if (c.explanation_text) parts.push(el('pre', { class: 'explanation', text: c.explanation_text }));
    return parts;
  }

  // ------------------------------------------------------------------ history
  function renderHistory() {
    const list = $('#history');
    if (!state.history.length) return;
    list.replaceChildren(...state.history.map((entry) => {
      const c = entry.data.comparison;
      const band = BANDS[c.band] ? c.band : 'INCONCLUSIVE';
      const label = entry.sample ? `${entry.sample.dataset_label} #${entry.sample.id}` : entry.names.questioned;
      const time = entry.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const button = el('button', { type: 'button', 'aria-label': `Show run ${entry.id}: ${BANDS[band].label}` }, [
        el('div', { class: 'h-top' }, [
          el('span', { class: `h-tag ${band}`, text: BANDS[band].label }),
          el('span', { class: 'h-score', text: c.match_score === null ? '–' : fmt(c.match_score, 1) }),
        ]),
        el('div', { class: 'h-meta', text: label }),
        el('div', { class: 'h-meta', text: `${time} · ${entry.names.reference} vs ${entry.names.questioned}` }),
      ]);
      button.addEventListener('click', () => showEntry(entry));
      return el('li', { class: `history-item${entry.id === state.activeHistoryId ? ' active' : ''}` }, button);
    }));
  }

  // ------------------------------------------------------------------ keyboard
  function setupKeyboard() {
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); verify(); }
      if (e.altKey && e.code === 'KeyN') { e.preventDefault(); nextSample(); }
    });
  }

  document.querySelectorAll('.zone').forEach(setupZone);
  $('#verify').addEventListener('click', verify);
  setupPaste();
  setupKeyboard();
  loadHealth();
  loadSamples();
})();
