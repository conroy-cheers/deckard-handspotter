'use strict';

const { app, dialog } = require('electron');
const { execFile, spawn } = require('node:child_process');
const { promisify } = require('node:util');
const { mkdtemp, readdir, rm, writeFile } = require('node:fs/promises');
const { tmpdir } = require('node:os');
const { join, resolve } = require('node:path');

const execute = promisify(execFile);
const backend = join(__dirname, 'batch');

async function readPreview(output, index, mode) {
  const result = await execute(
    join(__dirname, 'preview-python'),
    [join(__dirname, 'preview.py'), output, String(index), mode],
    { maxBuffer: 64 * 1024 * 1024 },
  );
  return JSON.parse(result.stdout);
}

function registerBatch(handle, window, projectDefaults, defaults) {
  let output,
    child,
    busy = false;
  let cachedSchema;
  async function schema() {
    if (!cachedSchema) cachedSchema = JSON.parse((await execute(backend, ['--studio-schema'])).stdout);
    return cachedSchema;
  }
  async function resolveJob(job, directory) {
    if (!job || typeof job !== 'object' || Array.isArray(job))
      throw new Error('Expected a render job object');
    const fields = await schema();
    const result = { ...job };
    for (const [key, value] of Object.entries(job)) {
      if (!Object.hasOwn(fields, key)) throw new Error('Unknown job setting: ' + key);
      const field = fields[key];
      if (['file', 'directory'].includes(field.kind)) {
        result[key] = field.multiple ? value.map((v) => resolve(directory, v)) : resolve(directory, value);
      }
    }
    return result;
  }
  handle('studio:batch-fields', async () => {
    const fields = await schema();
    try {
      return { fields, project: projectDefaults(), packagedFoundation: defaults.foundation_scenes };
    } catch (error) {
      return {
        fields,
        project: defaults,
        packagedFoundation: defaults.foundation_scenes,
        error: error.message,
      };
    }
  });
  handle('studio:garments', async (path) => {
    const directory = path || (await schema()).garment_library.default;
    if (!directory) return [];
    const files = await readdir(directory, { withFileTypes: true });
    return files
      .filter((file) => (file.isFile() || file.isSymbolicLink()) && file.name.endsWith('.json'))
      .map((file) => file.name.slice(0, -5))
      .sort();
  });
  handle('studio:batch-path', async ({ kind, multiple }) => {
    const properties = [kind === 'directory' ? 'openDirectory' : 'openFile'];
    if (multiple) properties.push('multiSelections');
    return (await dialog.showOpenDialog(window, { properties })).filePaths;
  });
  handle('studio:batch-open', async () => {
    const pick = await dialog.showOpenDialog(window, {
      properties: ['openFile'],
      filters: [{ name: 'Render job', extensions: ['json'] }],
    });
    if (pick.canceled) return;
    const path = pick.filePaths[0];
    const resolved = await execute(backend, ['--studio-resolve-job', '--job', path, '--out', tmpdir()]);
    return JSON.parse(resolved.stdout);
  });
  handle('studio:batch-save', async (job) => {
    const pick = await dialog.showSaveDialog(window, {
      title: 'Save render job',
      defaultPath: 'render-job.json',
    });
    if (!pick.canceled)
      await writeFile(pick.filePath, JSON.stringify(await resolveJob(job, process.cwd()), null, 2) + '\n');
    return pick.filePath;
  });
  let revision = 0;
  let session,
    pending,
    queue = Promise.resolve();
  function startSession() {
    const proc = spawn(backend, ['--studio-session'], { stdio: ['pipe', 'pipe', 'pipe'] });
    session = proc;
    let diagnostic = '';
    proc.stderr.on('data', (chunk) => {
      diagnostic = (diagnostic + chunk).slice(-2048);
    });
    const fail = (error) => {
      if (session !== proc) return;
      session = undefined;
      pending?.reject(error);
      pending = undefined;
      proc.stdin.end();
    };
    proc.on('error', fail);
    proc.stdin.on('error', fail);
    proc.on('close', () => fail(new Error(diagnostic.trim() || 'Viewport session closed')));
    require('node:readline')
      .createInterface({ input: proc.stdout })
      .on('line', (line) => {
        if (!pending || session !== proc) return;
        try {
          const reply = JSON.parse(line);
          if (reply.error) pending.reject(new Error(reply.error));
          else pending.resolve(reply);
        } catch (error) {
          pending.reject(error);
        }
        pending = undefined;
      });
  }
  function closeViewport() {
    revision++;
    session?.stdin.end();
  }
  app.once('before-quit', closeViewport);
  window.once('closed', closeViewport);
  function sessionRequest(message) {
    if (!session) startSession();
    return new Promise((resolve, reject) => {
      pending = { resolve, reject };
      session.stdin.write(JSON.stringify(message) + '\n');
    });
  }
  function enqueue(task) {
    const request = queue.then(task);
    queue = request.catch(() => {});
    return request;
  }
  handle('studio:viewport', async ({ job, index, mode }) => {
    const current = ++revision;
    if (job === null) return;
    if (!Number.isSafeInteger(index) || index < 0) throw new Error('Invalid viewport sample');
    const settings = await resolveJob(job, process.cwd());
    return enqueue(() => (current === revision ? sessionRequest({ job: settings, index, mode }) : undefined));
  });
  handle('studio:detector-result', (version) =>
    enqueue(async () => {
      if (!session || !Number.isSafeInteger(version)) throw new Error('Invalid detector capture');
      return sessionRequest({ finish: version });
    }),
  );
  window.on('closed', () => {
    if (child?.pid) {
      try {
        process.kill(-child.pid, 'SIGTERM');
      } catch (e) {
        if (e.code !== 'ESRCH') throw e;
      }
    }
  });
  handle('studio:batch-run', async (job) => {
    if (busy) throw new Error('A batch is already running');
    busy = true;
    let stage;
    try {
      const pick = await dialog.showSaveDialog(window, {
        title: 'Choose a new output directory',
        defaultPath: 'rendered-batch',
        buttonLabel: 'Render here',
      });
      if (pick.canceled) return;
      stage = await mkdtemp(join(tmpdir(), 'studio-job-'));
      const recipe = join(stage, 'job.json');
      await writeFile(recipe, JSON.stringify(await resolveJob(job, process.cwd())));
      await new Promise((done, fail) => {
        child = spawn(backend, ['--job', recipe, '--out', pick.filePath], {
          detached: true,
          stdio: ['ignore', 'ignore', 'pipe'],
        });
        let error = '';
        child.stderr.on('data', (bytes) => {
          error = (error + bytes).slice(-4096);
        });
        child.on('error', fail);
        child.on('close', (code) =>
          code === 0 ? done() : fail(new Error(error.trim() || 'Rendering stopped')),
        );
      });
      output = pick.filePath;
      return output;
    } finally {
      child = undefined;
      busy = false;
      if (stage) await rm(stage, { recursive: true, force: true });
    }
  });
  handle('studio:batch-preview', async ({ index, mode }) => {
    if (!output || !Number.isSafeInteger(index) || index < 0 || !['detector', 'image', 'mask'].includes(mode))
      throw new Error('Invalid batch preview');
    return readPreview(output, index, mode);
  });
}

module.exports = { registerBatch };
