'use strict';

const { app, BrowserWindow, dialog, ipcMain, Menu, net, protocol } = require('electron');
const { execFile, spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const { join, resolve } = require('node:path');
const { createInterface } = require('node:readline');
const { pathToFileURL } = require('node:url');

const home = 'labeler://app';
const launcher = home + '/launcher.html';
const cli = process.env.FRAME_HAND_LABELER_CLI;
const initialWorkspace = process.argv[process.argv.indexOf(__filename) + 1];
const tokenName = 'FRAME_HAND_LABELER_DESKTOP_TOKEN';
const assets = new Set(['/launcher.html', '/launcher.js', '/launcher.css']);
const jobs = new Set();
let window;
let active;
let busy = false;
let quitting = false;

app.setName('Frame Hand Labeler');
app.setPath('userData', join(app.getPath('appData'), 'frame-hand-labeler'));
if (process.env.WAYLAND_DISPLAY) app.commandLine.appendSwitch('ozone-platform', 'wayland');
app.commandLine.appendSwitch('disable-background-networking');

protocol.registerSchemesAsPrivileged([{
  scheme: 'labeler',
  privileges: { standard: true, secure: true, supportFetchAPI: true },
}]);

async function serve(request) {
  const url = new URL(request.url);
  if (url.host === 'app' && assets.has(url.pathname) && request.method === 'GET') {
    const file = await net.fetch(pathToFileURL(join(__dirname, url.pathname.slice(1))).href);
    return new Response(file.body, {
      headers: {
        'Content-Type': file.headers.get('Content-Type'),
        'Content-Security-Policy': "default-src 'self'; base-uri 'none'; frame-ancestors 'none'",
      },
    });
  }
  const server = active;
  if (url.host !== server?.host) return new Response(null, { status: 404 });
  const headers = { 'X-Labeler-Token': server.token };
  const options = { method: request.method, headers, redirect: 'error' };
  if (request.method === 'POST') {
    headers['Content-Type'] = request.headers.get('Content-Type');
    options.body = await request.text();
  } else if (request.method !== 'GET') {
    return new Response(null, { status: 405 });
  }
  try {
    const response = await fetch(server.origin + url.pathname + url.search, options);
    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type'),
        'Cache-Control': 'no-store',
        'Content-Security-Policy': "default-src 'self'; base-uri 'none'; frame-ancestors 'none'",
      },
    });
  } catch {
    return new Response('The workspace service is unavailable.', { status: 502 });
  }
}

function startWorkspace(path) {
  return new Promise((done, fail) => {
    const token = randomBytes(32).toString('hex');
    const child = spawn(cli, ['serve', path, '--port', '0'], {
      env: { ...process.env, [tokenName]: token }, stdio: ['pipe', 'pipe', 'pipe'],
    });
    jobs.add(child);
    child.stdin.on('error', () => {});
    let stderr = '';
    child.stderr.on('data', (data) => { stderr = (stderr + data).slice(-4096); });
    const lines = createInterface({ input: child.stdout });
    const timeout = setTimeout(() => {
      child.kill();
      fail(new Error('The workspace service did not start.'));
    }, 30000);
    child.on('error', (error) => {
      jobs.delete(child);
      clearTimeout(timeout);
      fail(error);
    });
    child.on('exit', () => {
      jobs.delete(child);
      clearTimeout(timeout);
      fail(new Error(stderr.trim() || 'The workspace service stopped.'));
      if (active?.child === child && !quitting) {
        active = undefined;
        window.loadURL(launcher).catch(console.error);
        dialog.showErrorBox('Workspace closed', stderr.trim() || 'The workspace service stopped.');
      }
    });
    lines.once('line', (line) => {
      const port = /^Open http:\/\/127\.0\.0\.1:(\d+) to label /.exec(line)?.[1];
      if (!port) {
        child.kill();
        fail(new Error('Invalid response from the workspace service.'));
        return;
      }
      clearTimeout(timeout);
      done({ path, child, token, origin: 'http://127.0.0.1:' + port,
             host: 'workspace-' + port + '-' + randomBytes(8).toString('hex') });
    });
  });
}

function runCLI(args) {
  return new Promise((done, fail) => {
    const child = execFile(cli, args, (error, stdout, stderr) => {
      jobs.delete(child);
      if (error) fail(new Error(stderr.trim() || error.message));
      else done(stdout.trim());
    });
    jobs.add(child);
  });
}

async function loadWorkspace(path) {
  const next = await startWorkspace(resolve(path));
  const previous = active;
  active = next;
  try {
    await window.loadURL('labeler://' + next.host + '/');
  } catch (error) {
    if (active === next) active = previous;
    next.child.stdin.end();
    throw error;
  }
  previous?.child.stdin.end();
}

async function transaction(fn) {
  if (busy || quitting) throw new Error('Another workspace operation is in progress.');
  busy = true;
  try { return await fn(); }
  finally { busy = false; }
}

async function choose(kind) {
  const options = {
    workspace: { title: 'Open labeling workspace', properties: ['openDirectory'] },
    recording: { title: 'Select Frame recording', properties: ['openDirectory'] },
    archive: { title: 'Select HOT3D clip', properties: ['openFile'], filters: [{ name: 'HOT3D clips', extensions: ['tar'] }] },
    rig: { title: 'Select Frame warp calibration', properties: ['openFile'], filters: [{ name: 'Camera rigs', extensions: ['json'] }] },
  };
  if (!Object.hasOwn(options, kind)) throw new Error('Unknown file selection.');
  const pick = await dialog.showOpenDialog(window, options[kind]);
  return pick.filePaths[0] ?? null;
}

async function openWorkspace() {
  const path = await choose('workspace');
  if (path) await loadWorkspace(path);
}

async function importWorkspace(settings) {
  const pick = await dialog.showSaveDialog(window, {
    title: 'Create labeling workspace',
    buttonLabel: 'Create workspace',
    defaultPath: 'labeling-workspace',
  });
  if (pick.canceled || !pick.filePath) return;
  const args = ['ingest', settings.source, pick.filePath];
  for (const key of ['start', 'duration', 'every', 'hz']) {
    if (settings[key]) args.push('--' + key, settings[key]);
  }
  if (settings.rig) args.push('--warp-rig', settings.rig);
  for (const channel of settings.channels ?? []) args.push('--channel-map', channel);
  await runCLI(args);
  await loadWorkspace(pick.filePath);
}

async function exportLabels() {
  if (!active) throw new Error('Open a workspace first.');
  const text = await runCLI(['export', active.path]);
  await dialog.showMessageBox(window, { message: 'Labels exported', detail: text });
}

function setupMenu() {
  const action = (fn) => () => transaction(fn).catch((error) => dialog.showErrorBox('Frame Hand Labeler', error.message));
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'File', submenu: [
      { label: 'Open workspace…', accelerator: 'Ctrl+O', click: action(openWorkspace) },
      { label: 'Import recording or clip…', accelerator: 'Ctrl+I', click: action(() => window.loadURL(launcher)) },
      { label: 'Export labels', accelerator: 'Ctrl+E', click: action(exportLabels) },
      { type: 'separator' },
      { role: 'quit' },
    ] },
    { role: 'editMenu' },
    { role: 'viewMenu', submenu: [{ role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }] },
  ]));
}

app.on('before-quit', () => {
  quitting = true;
  for (const child of jobs) {
    if (child === active?.child) child.stdin.end();
    else child.kill('SIGINT');
  }
});
app.on('window-all-closed', () => app.quit());

app.whenReady().then(async () => {
  protocol.handle('labeler', serve);
  window = new BrowserWindow({
    title: 'Frame Hand Labeler', width: 1200, height: 820, minWidth: 640, minHeight: 500,
    webPreferences: {
      preload: join(__dirname, 'preload.cjs'), contextIsolation: true, sandbox: true, nodeIntegration: false,
    },
  });
  const contents = window.webContents;
  contents.on('will-navigate', (event, url) => {
    if (url !== launcher && url !== 'labeler://' + active?.host + '/') event.preventDefault();
  });
  contents.setWindowOpenHandler(() => ({ action: 'deny' }));
  function handle(name, run, exclusive = true) {
    ipcMain.handle('labeler:' + name, (event, value) => {
      if (event.senderFrame !== contents.mainFrame || event.senderFrame?.url !== launcher) {
        throw new Error('Unexpected workspace command sender.');
      }
      return exclusive ? transaction(() => run(value)) : run(value);
    });
  }
  handle('choose', choose);
  handle('open', openWorkspace);
  handle('import', importWorkspace);
  handle('resume', () => active && window.loadURL('labeler://' + active.host + '/'));
  handle('hasWorkspace', () => Boolean(active), false);
  setupMenu();
  await window.loadURL(launcher);
  if (initialWorkspace) {
    await transaction(() => loadWorkspace(initialWorkspace)).catch((error) => dialog.showErrorBox('Cannot open workspace', error.message));
  }
}).catch((error) => {
  console.error(error);
  app.quit();
});
