import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import http from 'node:http';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const [stage, assets, three, chromium, foundation] = process.argv.slice(2);
const source = fileURLToPath(new URL('.', import.meta.url));
const session = chromium === '--studio-session';
const job = session ? {viewport:0, eyes:[], count:0, wardrobe_mode:true} :
  JSON.parse(await readFile(join(stage, 'job.json')));
const clothDependencies=process.env.ANYHAND_CLOTH_JS;
if (job.wardrobe_mode && !clothDependencies) throw new Error('Use the anyhand-pbr shell for the SOMA-X wardrobe');
const foundationFiles = new Set(job.foundation_files);
const viewport = job.viewport != null;
const frameIndices = Array.from({length:job.count}, (_,index)=>index);
const inputs = new Set(['/job.json',
  ...frameIndices.map(frame=>`/${job.sample ?? frame}.${job.binary_surfaces ? 'bin' : 'json'}`)]);
const outputs = new Set(frameIndices.flatMap(frame=>job.eyes.map((_,eye)=>`/output/${frame}-${eye}.bin`)));
const contentType = {
  '.html': 'text/html',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.json': 'application/json',
  '.gltf': 'model/gltf+json',
  '.wasm': 'application/wasm',
};

const html = `<!doctype html><meta charset="utf-8"><title>AnyHand rendering</title>
<script type="importmap">{"imports":{"three":"/three/build/three.module.js","three/webgpu":"/three/build/three.webgpu.js","three/tsl":"/three/build/three.tsl.js"}}</script>
<script type="module" src="/pbr_render.mjs"></script>`;

function input(path) {
  if (/^\/(pbr_render|pbr_scene|pbr_motion|pbr_sensor|pbr_photometry)\.mjs$/.test(path)) return join(source, path.slice(1));
  if (job.wardrobe_mode && /^\/(pbr_cloth|skin_collider)\.mjs$/.test(path)) return join(source,path.slice(1));
  if (job.wardrobe_mode && path === '/skin-collider-topology.bin') return process.env.ANYHAND_SKIN_TOPOLOGY;
  if (job.wardrobe_mode && path === '/skin-collider.wasm') return process.env.ANYHAND_SKIN_KERNEL;
  if (job.wardrobe_mode && /^\/cloth-deps\/mesh-bvh\.mjs$/.test(path))
    return join(clothDependencies,path.slice(12));
  if (viewport && path === '/viewport.mjs') return join(source, 'viewport.mjs');
  if (viewport && path === '/three/examples/jsm/controls/OrbitControls.js') return join(three,path.slice(7));
  if (inputs.has(path)) return join(stage, path.slice(1));
  if (/^\/assets\/[\w-]+\.(jpg|png|hdr)$/.test(path)) return join(assets, path.slice(8));
  if (path.startsWith('/foundation/') && foundationFiles.has(path.slice(12)))
    return join(foundation, path.slice(12));
  if (/^\/three\/(build\/three\.(module|core|webgpu|tsl)\.js|examples\/jsm\/(loaders\/(HDR|GLTF)Loader|lights\/RectAreaLight(Uniforms|Textures)Lib|utils\/(BufferGeometryUtils|SkeletonUtils))\.js)$/.test(path))
    return join(three, path.slice(7));
  return null;
}

let finish;
let timeout;
const completion = new Promise(resolve => { finish = resolve; });
const server = http.createServer(async (request, response) => {
  try {
    const path = new URL(request.url, 'http://127.0.0.1').pathname;
    if (request.method === 'GET') {
      if (path === '/') {
        response.writeHead(200, {'Content-Type': 'text/html'}).end(html);
        return;
      }
      const versioned = session && /^\/v\/([1-9]\d*)\/(job\.json|\d+\.(?:bin|json)|foundation\/[\w-]+\/[\w.-]+)$/.exec(path);
      let file;
      if (versioned) {
        const [, version, name] = versioned;
        const directory = join(stage, version);
        if (name.startsWith('foundation/')) {
          const manifest = JSON.parse(await readFile(join(directory, 'job.json')));
          const asset = name.slice('foundation/'.length);
          if (manifest.foundation_files.includes(asset))
            file = join(await readFile(join(directory, 'foundation-root'), 'utf8'), asset);
        } else file = join(directory, name);
      } else if (!session || !inputs.has(path)) file = input(path);
      if (file) {
        const suffix = file.slice(file.lastIndexOf('.'));
        const data = await readFile(file);
        response.writeHead(200, {'Content-Type': contentType[suffix] || 'application/octet-stream'}).end(data);
        return;
      }
    }
    const capture = session && request.method === 'POST' &&
      /^\/v\/([1-9]\d*)\/output\/0-([0-3])\.bin$/.exec(path);
    if (capture) {
      const directory=join(stage,capture[1]);
      const manifest=JSON.parse(await readFile(join(directory,'job.json')));
      const size=manifest.sensor_sizes[Number(capture[2])];
      if (!manifest.capture || request.headers['x-capture-token'] !== manifest.capture.token ||
          !size || Number(request.headers['content-length']) !== 64+160*128) {
        response.writeHead(403).end();
        return;
      }
      await pipeline(request,createWriteStream(join(directory,`0-${capture[2]}.bin`)));
      response.writeHead(204).end();
      return;
    }
    if (!viewport && request.method === 'POST' && outputs.delete(path)) {
      await pipeline(request, createWriteStream(join(stage, path.slice(8))));
      timeout?.refresh();
      response.writeHead(204).end();
      return;
    }
    if (!viewport && request.method === 'POST' && path === '/result') {
      const data = [];
      for await (const chunk of request) data.push(chunk);
      const result = JSON.parse(Buffer.concat(data));
      await writeFile(join(stage, 'result.json'), JSON.stringify(result));
      response.writeHead(204).end(() => finish(result.error || null));
      return;
    }
    response.writeHead(404).end();
  } catch (error) {
    response.writeHead(500).end(String(error));
    finish(String(error));
  }
});

await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
if (viewport) {
  console.log('STUDIO_VIEWPORT http://127.0.0.1:'+server.address().port+'/');
  // Electron owns this pipe. Release the scene server if its host disappears.
  process.stdin.on('end', () => { server.close(); server.closeAllConnections(); });
  process.stdin.resume();
  await new Promise(resolve => server.once('close', resolve));
} else {
  const env = {...process.env};
  delete env.DISPLAY;
  delete env.WAYLAND_DISPLAY;
  const angle = env.ANYHAND_SOFTWARE_RENDER === '1'
    ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    : ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface', '--disable-software-rasterizer'];
  const browser = spawn(chromium, [
    '--headless', '--ozone-platform=headless', '--use-gl=angle', ...angle,
    '--disable-background-networking', '--no-first-run', `--user-data-dir=${join(stage, 'chromium')}`,
    ...(job.wardrobe_mode==='drape' ? ['--enable-unsafe-webgpu'] : []),
    `http://127.0.0.1:${server.address().port}/`,
  ], {env, stdio: ['ignore', 'ignore', 'pipe']});

  let stderr = '';
  browser.stderr.on('data', chunk => { stderr = (stderr+chunk).slice(-6000); });
  const exited = new Promise(resolve => {
    browser.once('error', error => finish(String(error)));
    browser.once('close', code => {
      finish(`Chromium exited before rendering finished (${code})`);
      resolve();
    });
  });
  timeout = setTimeout(() => finish('Chromium stopped producing camera views'), 600_000);
  const error = await completion;
  clearTimeout(timeout);
  browser.kill();
  await exited;
  server.closeAllConnections();
  server.close();
  if (error) {
    console.error(error);
    console.error(stderr);
    process.exitCode = 1;
  }
}
