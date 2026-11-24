const api = window.anyhand;
const el = (id) => document.getElementById(id);
const primary = [
  'rig_json',
  'calibration',
  'openvr_calibration',
  'soma_template',
  'body_bank',
  'source',
  'topology',
  'environment',
  'foundation_scenes',
  'soma_wardrobe',
  'seed',
  'count',
];
const groups = {
  camera: { rig_json: 'Frame four-camera rig', calibration: 'Stereo calibration' },
  geometry: {
    soma_template: 'SOMA-X pose template',
    source: 'Pose bank and topology',
    body_bank: 'Baked bodies',
  },
};
const names = {
  soma_wardrobe: 'Garment',
  rig_json: 'Rig JSON',
  calibration: 'Calibration file',
  soma_template: 'Template dataset',
  source: 'Pose banks (one path per line)',
  topology: 'Hand topology',
  count: 'Samples',
  seed: 'Batch seed',
  randomize_poses: 'Generate new reaches per sample',
  pose_edit: 'Pose override (index,yaw,pitch,roll)',
  native_soma: 'Evaluate native SOMA-X poses',
};
const label = (key) => names[key] || (key[0].toUpperCase() + key.slice(1)).replaceAll('_', ' ');

function initBatch() {
  let packagedFoundation,
    fields,
    sample,
    previewRevision = 0;
  const inputs = new Map(),
    modes = {};
  const status = (message) => {
    el('batch-status').textContent = message;
  };
  const guarded = (fn) => async () => {
    try {
      await fn();
    } catch (error) {
      status(error.message);
    }
  };

  let liveRevision = 0,
    timer,
    origin;
  let frame = el('live-frame'),
    pendingFrame,
    cameraState,
    expectedVersion;
  const liveStatus = (text) => {
    el('live-status').textContent = text;
  };
  window.addEventListener('message', async (event) => {
    if (
      event.source === frame.contentWindow &&
      frame.src &&
      event.origin === new URL(frame.src).origin &&
      event.data?.studioViewport === 'camera'
    ) {
      cameraState = event.data.camera;
      return;
    }
    if (event.origin !== origin || ![pendingFrame?.contentWindow, frame.contentWindow].includes(event.source))
      return;
    const { studioViewport, version, setup, message } = event.data || {};
    if (
      studioViewport === 'error' &&
      (version === expectedVersion || event.source === pendingFrame?.contentWindow)
    ) {
      pendingFrame?.remove();
      pendingFrame = undefined;
      liveStatus(message);
      return;
    }
    if (studioViewport !== 'ready' || version !== expectedVersion) return;
    if (pendingFrame && event.source === pendingFrame.contentWindow) {
      frame.remove();
      frame = pendingFrame;
      pendingFrame = undefined;
      frame.id = 'live-frame';
      frame.classList.remove('pending');
    }
    const detector = el('live-mode').value === 'detector';
    if (detector) {
      const current = liveRevision;
      try {
        const result = await api.detectorResult(version);
        if (current !== liveRevision) return;
        drawViews(el('live-detectors'), result.views, false);
      } catch (error) {
        if (current === liveRevision) liveStatus(error.message);
        return;
      }
    }
    frame.hidden = detector;
    el('live-detectors').hidden = !detector;
    showSetup(setup);
    liveStatus('');
  });
  function showSetup(setup) {
    const source = { recorded: 'recorded room', override: 'job selection', generated: 'generated room' };
    el('resolved-setup').textContent = setup
      ? [
          setup.environment + ' (' + source[setup.environment_source] + ')',
          setup.empty_room
            ? 'Empty room'
            : setup.added_garment
              ? setup.added_garment + ' (' + setup.cloth_mode + ')'
              : 'No added garment',
        ].join(' · ')
      : '';
  }
  function jobProblem(job) {
    if (!job[modes.camera.value] || !job[modes.geometry.value] || job.seed === undefined)
      return 'Choose camera and geometry inputs and enter a render seed to render the scene.';
    if (job.environment?.startsWith('foundation-') && !job.foundation_scenes && !packagedFoundation)
      return (
        'Choose the Foundation scenes directory containing the ' +
        job.environment.slice('foundation-'.length) +
        ' folder.'
      );
    if (job.source && !job.topology) return 'Choose the topology for the pose bank.';
    if (job.native_soma && job.rig_json && !job.openvr_calibration)
      return 'Choose the matching OpenVR calibration.';
    return '';
  }
  function scheduleViewport(event) {
    const current = ++liveRevision;
    showSetup(null);
    clearTimeout(timer);
    pendingFrame?.remove();
    pendingFrame = undefined;
    origin = undefined;
    expectedVersion = undefined;
    const job = settings();
    const selector = el('live-index');
    const count = job.count ?? 24;
    selector.max = String(Math.max(0, count - 1));
    let issue = jobProblem(job);
    if (
      !issue &&
      (!Number.isInteger(selector.valueAsNumber) ||
        selector.valueAsNumber < 0 ||
        selector.valueAsNumber >= count)
    )
      issue = 'Choose a sample inside this batch.';
    liveStatus(issue || 'Updating scene…');
    if (issue) {
      el('live-detectors').hidden = true;
      frame.hidden = true;
      api.viewport(null, 0).catch(console.error);
      return;
    }
    timer = setTimeout(
      async () => {
        try {
          const mode = el('live-mode').value;
          const result = await api.viewport(job, selector.valueAsNumber, mode);
          if (current !== liveRevision || !result) return;
          expectedVersion = result.version;
          const url = result.url + '?version=' + result.version;
          origin = new URL(url).origin;
          if (frame.src && new URL(frame.src).origin === origin) {
            frame.contentWindow.postMessage({ version: result.version }, origin);
            return;
          }
          pendingFrame = frame.cloneNode(false);
          pendingFrame.removeAttribute('id');
          pendingFrame.classList.add('pending');
          pendingFrame.hidden = false;
          pendingFrame.src = cameraState ? url + '#' + encodeURIComponent(JSON.stringify(cameraState)) : url;
          frame.after(pendingFrame);
        } catch (error) {
          if (current === liveRevision) {
            frame.hidden = true;
            el('live-detectors').hidden = true;
            liveStatus(error.message);
          }
        }
      },
      event?.type === 'input' && event.target.tagName !== 'SELECT' ? 250 : 0,
    );
  }
  function updateModes() {
    for (const [key, input] of inputs) {
      const camera = Object.hasOwn(groups.camera, key);
      const source = Object.hasOwn(groups.geometry, key);
      input.closest('.setting').hidden =
        (camera && key !== modes.camera.value) ||
        (source && key !== modes.geometry.value) ||
        (key === 'soma_wardrobe' &&
          !(inputs.get('native_soma').checked || modes.geometry.value === 'body_bank')) ||
        key === 'cloth_mode' ||
        (key === 'randomize_poses' && modes.geometry.value !== 'soma_template') ||
        (key === 'topology' && modes.geometry.value !== 'source') ||
        (key === 'openvr_calibration' &&
          (modes.camera.value !== 'rig_json' || !inputs.get('native_soma').checked));
    }
  }
  function addMode(id, title) {
    const select = document.createElement('select');
    for (const [key, text] of Object.entries(groups[id])) select.add(new Option(text, key));
    const wrap = document.createElement('label');
    wrap.className = 'setting';
    wrap.append(title, select);
    el('primary-settings').append(wrap);
    modes[id] = select;
    select.onchange = () => {
      if (id === 'geometry') inputs.get('native_soma').checked = select.value === 'soma_template';
      updateModes();
    };
  }
  function addField(key, field) {
    const wrapper = document.createElement('div'),
      caption = document.createElement('label'),
      row = document.createElement('div');
    wrapper.className = 'setting';
    row.className = 'row';
    const input = document.createElement(
      field.choices || field.suggestions ? 'select' : field.multiple ? 'textarea' : 'input',
    );
    if (field.suggestions) {
      input.add(new Option('No added garment', ''));
      for (const name of field.suggestions) input.add(new Option(name, name));
    } else if (field.choices) {
      input.add(
        new Option(key === 'environment' ? 'Use recorded room when available' : 'Renderer default', ''),
      );
      for (const choice of field.choices) input.add(new Option(String(choice), String(choice)));
    } else if (field.kind === 'flag') input.type = 'checkbox';
    else if (field.kind === 'integer') input.type = 'number';
    input.id = 'setting-' + key;
    caption.htmlFor = input.id;
    caption.textContent = label(key);
    if (field.help) {
      input.title = field.help;
      caption.title = field.help;
    }
    if (field.default != null && field.kind !== 'flag') input.placeholder = String(field.default);
    inputs.set(key, input);
    row.append(input);
    if (['file', 'directory'].includes(field.kind)) {
      const browse = document.createElement('button');
      browse.type = 'button';
      browse.textContent = 'Browse…';
      browse.setAttribute('aria-label', 'Browse for ' + label(key));
      browse.onclick = guarded(async () => {
        const paths = await api.pickPath(field.kind, field.multiple);
        if (paths.length) {
          input.value = paths.join('\n');
          if (key === 'garment_library') refreshGarments();
          scheduleViewport();
        }
      });
      row.append(browse);
    }
    wrapper.append(caption, row);
    el(primary.includes(key) ? 'primary-settings' : 'advanced-settings').append(wrapper);
  }
  let libraryRevision = 0;
  async function refreshGarments() {
    const revision = ++libraryRevision;
    const picker = inputs.get('soma_wardrobe');
    const library = inputs.get('garment_library').value;
    let names = [];
    try {
      names = await api.garments(library);
    } catch (error) {
      if (revision === libraryRevision) status(error.message);
    }
    if (revision !== libraryRevision) return;
    const selected = picker.value;
    const options = new Set([
      ...fields.soma_wardrobe.suggestions,
      ...(names.length ? ['garmentcode-balanced', ...names] : []),
      ...(selected ? [selected] : []),
    ]);
    picker.replaceChildren(new Option('No added garment', ''));
    for (const name of options) picker.add(new Option(name, name));
    picker.value = selected;
  }
  function fill(job) {
    for (const [key, input] of inputs) {
      const value = job[key];
      if (key === 'soma_wardrobe' && value && !Array.from(input.options).some((o) => o.value === value))
        input.add(new Option(value, value));
      if (fields[key].kind === 'flag') input.checked = value === true;
      else input.value = Array.isArray(value) ? value.join('\n') : (value ?? '');
    }
    modes.camera.value = job.calibration ? 'calibration' : 'rig_json';
    modes.geometry.value = job.source ? 'source' : job.body_bank ? 'body_bank' : 'soma_template';
    updateModes();
    refreshGarments();
  }
  function settings() {
    const job = {};
    for (const [key, input] of inputs) {
      if (input.closest('.setting').hidden) continue;
      const field = fields[key],
        value = input.value.trim();
      if (field.kind === 'flag') {
        if (input.checked || field.negatable) job[key] = input.checked;
      } else if (value)
        job[key] = field.multiple
          ? value
              .split('\n')
              .map((v) => v.trim())
              .filter(Boolean)
          : field.kind === 'integer'
            ? Number(value)
            : value;
    }
    return job;
  }
  function drawViews(grid, views, boxes) {
    grid.replaceChildren();
    for (const view of views) {
      const fig = document.createElement('figure'),
        frame = document.createElement('div'),
        img = document.createElement('img'),
        caption = document.createElement('figcaption');
      frame.className = 'batch-frame';
      img.src = 'data:image/png;base64,' + view.png;
      img.alt = view.stream;
      frame.append(img);
      if (boxes)
        for (const [side, [x, y, r, b]] of Object.entries(view.boxes)) {
          const box = document.createElement('span');
          box.className = 'hand-box ' + side;
          box.textContent = side;
          Object.assign(box.style, {
            left: (100 * x) / view.width + '%',
            top: (100 * y) / view.height + '%',
            width: (100 * (r - x)) / view.width + '%',
            height: (100 * (b - y)) / view.height + '%',
          });
          frame.append(box);
        }
      caption.textContent = view.stream + ' · ' + view.width + ' × ' + view.height;
      fig.append(caption, frame);
      grid.append(fig);
    }
  }
  const draw = () => drawViews(el('batch-views'), sample.views, el('batch-boxes').checked);
  async function preview() {
    const revision = ++previewRevision;
    const data = await api.previewBatch(Number(el('batch-index').value), el('batch-mode').value);
    if (revision !== previewRevision) return;
    sample = data;
    el('batch-index').max = String(data.count - 1);
    for (const opt of el('batch-mode').options)
      opt.disabled = opt.value === 'detector' ? !data.detector : !data.images;
    el('batch-result').hidden = false;
    draw();
  }
  async function setup() {
    const result = await api.batchFields();
    fields = result.fields;
    packagedFoundation = result.packagedFoundation;
    addMode('camera', 'Camera setup');
    addMode('geometry', 'Geometry source');
    for (const key of [...primary, ...Object.keys(fields).filter((key) => !primary.includes(key))])
      addField(key, fields[key]);
    const nextSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
    fill({ ...result.project, seed: result.project.seed ?? nextSeed() });
    el('reroll').onclick = () => {
      inputs.get('seed').value = String(nextSeed());
      scheduleViewport();
    };
    inputs.get('foundation_scenes').placeholder = packagedFoundation || '';
    if (result.error) status(result.error);
    scheduleViewport();
    el('render-form').addEventListener('input', scheduleViewport);
    el('render-form').addEventListener('change', scheduleViewport);
    el('live-index').oninput = el('live-mode').onchange = scheduleViewport;
    inputs.get('garment_library').onchange = refreshGarments;
    inputs.get('native_soma').onchange = updateModes;
    for (const [key, other] of [
      ['wide_reach', 'hemisphere_reach'],
      ['hemisphere_reach', 'wide_reach'],
    ]) {
      inputs.get(key).onchange = () => {
        if (inputs.get(key).checked) inputs.get(other).checked = false;
      };
    }
    el('open-job').onclick = guarded(async () => {
      const job = await api.openJob();
      if (job) {
        fill(job);
        el('live-index').value = '0';
        scheduleViewport();
        status('Job loaded.');
      }
    });
    el('save-job').onclick = guarded(async () => {
      const path = await api.saveJob(settings());
      if (path) status('Saved ' + path);
    });
    el('render-form').onsubmit = async (event) => {
      event.preventDefault();
      el('batch-controls').disabled = true;
      el('open-job').disabled = true;
      try {
        const job = settings();
        const problem = jobProblem(job);
        if (problem) throw new Error(problem);
        status('Rendering batch…');
        const path = await api.renderBatch(job);
        if (!path) {
          status('Render canceled.');
          return;
        }
        el('batch-index').value = String(
          Math.min(Number(el('live-index').value) || 0, (job.count || 24) - 1),
        );
        el('batch-mode').value = job.calibration ? 'image' : 'detector';
        await preview();
        status('Rendered to ' + path);
        el('batch-result').scrollIntoView();
      } catch (error) {
        status(error.message);
      } finally {
        el('batch-controls').disabled = false;
        el('open-job').disabled = false;
      }
    };
    el('batch-index').onchange = el('batch-mode').onchange = guarded(preview);
    el('batch-boxes').onchange = draw;
  }
  setup().catch((error) => status(error.message));
}

initBatch();
