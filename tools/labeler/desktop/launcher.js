const get = (id) => document.getElementById(id);
const form = get('import');
const status = get('status');
let busy = false;

function update() {
  get('open').disabled = get('resume').disabled = busy;
  for (const control of form.elements) control.disabled = busy;
  get('create').disabled = busy || !get('source').value;
}

async function run(fn) {
  busy = true;
  status.textContent = '';
  update();
  try {
    await fn();
  } catch (error) {
    status.textContent = error.message.replace(/^Error invoking remote method '[^']+': Error: /, '');
  } finally {
    busy = false;
    update();
  }
}

get('open').onclick = () => run(() => window.labeler.open());
get('resume').onclick = () => run(() => window.labeler.resume());
get('kind').onchange = () => {
  get('source').value = '';
  get('hot3d').hidden = get('kind').value !== 'archive';
  update();
};
get('browse').onclick = () => run(async () => {
  const path = await window.labeler.choose(get('kind').value);
  if (path) get('source').value = path;
});
get('browse-rig').onclick = () => run(async () => {
  const path = await window.labeler.choose('rig');
  if (path) get('rig').value = path;
});
get('clear-rig').onclick = () => { get('rig').value = ''; };
get('sampling').onchange = () => {
  const interval = get('interval');
  const fixedRate = get('sampling').value === 'hz';
  interval.min = fixedRate ? '0.001' : '1';
  interval.step = fixedRate ? 'any' : '1';
  interval.value = fixedRate ? '10' : '1';
};
form.onsubmit = (event) => {
  event.preventDefault();
  const settings = {
    source: get('source').value,
    start: get('start').value,
    duration: get('duration').value,
    [get('sampling').value]: get('interval').value,
  };
  if (get('kind').value === 'archive') {
    settings.rig = get('rig').value;
    settings.channels = get('channels').value.split('\n').map((line) => line.trim()).filter(Boolean);
  }
  run(async () => {
    status.textContent = 'Importing synchronized views…';
    await window.labeler.importWorkspace(settings);
    status.textContent = '';
  });
};

get('resume').hidden = !(await window.labeler.hasWorkspace());
