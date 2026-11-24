'use strict';

const { app, BrowserWindow, ipcMain, net, protocol } = require('electron');
const { readFileSync } = require('node:fs');
const { homedir } = require('node:os');
const { dirname, isAbsolute, join, resolve, sep } = require('node:path');
const { pathToFileURL } = require('node:url');

const { registerBatch } = require('./batch.cjs');
const launchPage = 'studio://app/index.html';
const frontend = join(__dirname, 'frontend');
const configHome = isAbsolute(process.env.XDG_CONFIG_HOME || '')
  ? process.env.XDG_CONFIG_HOME
  : join(homedir(), '.config');
const configDir = join(configHome, 'anyhand-studio');
const defaultProject = join(configDir, 'project.json');
const starterProject = join(__dirname, 'starter-project.json');
const projectFields = {
  assets: 'SOMA-X assets directory',
  correctives: 'SOMA-X corrective checkpoint',
  template: 'Pose dataset',
  rig: 'Frame camera rig',
  openvr: 'OpenVR calibration',
  garments: 'GarmentCode library directory',
  foundation: 'Foundation scene directory',
};

function readProject(project, optional = false) {
  let raw;
  try {
    raw = readFileSync(project, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT' && optional) return null;
    if (error.code === 'ENOENT') throw new Error(`No project file at ${project}.`);
    throw new Error(`Could not read AnyHand project ${project}: ${error.message}`);
  }
  let files;
  try {
    files = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid AnyHand project ${project}: ${error.message}`);
  }
  if (!files || typeof files !== 'object' || Array.isArray(files)) {
    throw new Error(`Expected an object in AnyHand project ${project}.`);
  }
  return Object.fromEntries(
    Object.entries(files).map(([key, path]) => {
      if (!Object.hasOwn(projectFields, key)) {
        throw new Error(`Unknown AnyHand project field "${key}".`);
      }
      if (typeof path !== 'string' || !path.length) {
        throw new Error(`The AnyHand project ${key} must be a nonempty path.`);
      }
      return [key, resolve(dirname(project), path)];
    }),
  );
}

function serveAsset(request) {
  const url = new URL(request.url);
  const file = resolve(frontend, '.' + decodeURIComponent(url.pathname));
  if (request.method !== 'GET' || url.host !== 'app' || !file.startsWith(frontend + sep)) {
    return new Response(null, { status: 404 });
  }
  return net.fetch(pathToFileURL(file).href);
}

function main() {
  app.setName('AnyHand Studio');
  app.setPath('userData', join(configDir, 'electron'));
  protocol.registerSchemesAsPrivileged([
    {
      scheme: 'studio',
      privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
    },
  ]);
  const wayland = Boolean(process.env.WAYLAND_DISPLAY);
  const switches = {
    'enable-unsafe-webgpu': '',
    'use-gl': 'angle',
    'use-angle': wayland ? 'gles' : 'vulkan',
    'disable-background-networking': '',
  };
  if (wayland) {
    // Dawn can use the active EGL display for cloth compute on Wayland.
    switches['ozone-platform'] = 'wayland';
    switches['use-webgpu-adapter'] = 'opengles';
    switches['force-webgpu-compat'] = '';
  } else {
    switches['enable-features'] = 'Vulkan';
  }
  for (const [name, value] of Object.entries(switches)) app.commandLine.appendSwitch(name, value);

  const configured = process.env.ANYHAND_STUDIO_PROJECT;
  const defaults = JSON.parse(readFileSync(starterProject, 'utf8'));
  function projectDefaults() {
    const keys = {
      assets: 'soma_assets',
      correctives: 'soma_correctives',
      template: 'soma_template',
      rig: 'rig_json',
      openvr: 'openvr_calibration',
      garments: 'garment_library',
      foundation: 'foundation_scenes',
    };
    const overrides = readProject(resolve(configured || defaultProject), !configured) || {};
    return {
      ...defaults,
      ...(overrides.rig || overrides.openvr ? { rig_json: undefined, openvr_calibration: undefined } : {}),
      ...Object.fromEntries(Object.entries(overrides).map(([key, path]) => [keys[key], path])),
    };
  }
  app.on('window-all-closed', () => app.quit());
  app
    .whenReady()
    .then(async () => {
      protocol.handle('studio', serveAsset);
      const window = new BrowserWindow({
        title: 'AnyHand Studio',
        width: 1200,
        height: 820,
        minWidth: 650,
        minHeight: 500,
        icon: join(__dirname, 'icon.png'),
        autoHideMenuBar: true,
        webPreferences: {
          preload: join(__dirname, 'preload.cjs'),
          contextIsolation: true,
          sandbox: true,
          nodeIntegration: false,
        },
      });
      const { webContents } = window;
      webContents.on('will-navigate', (event, url) => {
        if (url !== launchPage) event.preventDefault();
      });
      webContents.setWindowOpenHandler(() => ({ action: 'deny' }));

      function handle(channel, run) {
        ipcMain.handle(channel, (event, value) => {
          if (event.senderFrame !== webContents.mainFrame || event.senderFrame?.url !== launchPage) {
            throw new Error('Unexpected Studio IPC sender');
          }
          return run(value);
        });
      }
      registerBatch(handle, window, projectDefaults, defaults);
      await window.loadURL(launchPage);
    })
    .catch((error) => {
      console.error(error);
      app.exit(1);
    });
}

main();
