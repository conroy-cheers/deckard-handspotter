import * as THREE from 'three';
import { OrbitControls } from '/three/examples/jsm/controls/OrbitControls.js';

export function showViewport(renderer) {
  const canvas = renderer.domElement;
  document.documentElement.style.cssText = 'height:100%;background:#182534;color:white;color-scheme:dark';
  document.body.style.cssText = 'margin:0;height:100%;overflow:hidden;font:14px system-ui';
  canvas.style.display = 'block';
  canvas.setAttribute('aria-label', 'Interactive batch scene');
  let scene, job;
  const camera = new THREE.PerspectiveCamera(55, 1, 0.015, 12);
  const controls = new OrbitControls(camera, canvas);
  controls.minPolarAngle = 0.001;
  controls.maxPolarAngle = Math.PI - 0.001;
  const select = document.createElement('select');
  select.setAttribute('aria-label', 'Viewport camera');
  select.style.cssText = 'position:absolute;top:10px;left:10px;padding:5px';
  select.add(new Option('Orbit camera', 'orbit'));
  document.body.replaceChildren(canvas, select);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.needsUpdate = true;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const draw = () => {
    if (scene && !renderer.getRenderTarget()) renderer.render(scene, camera);
  };
  controls.addEventListener('change', draw);
  function setView() {
    controls.enabled = select.value === 'orbit';
    if (controls.enabled) {
      camera.position.set(0.75, 0.2, 0.65);
      camera.up.set(0, 1, 0);
      controls.target.set(0, -0.22, -0.35);
      controls.update();
    } else {
      const view = job.eyes[Number(select.value)][0];
      camera.position.fromArray(view.position);
      camera.up.fromArray(view.up);
      const target = new THREE.Vector3().fromArray(view.look);
      const toPair = job.pair_from_reference?.[0];
      if (toPair) {
        const worldFromPair = new THREE.Matrix4().set(...toPair.flat()).invert();
        camera.position.applyMatrix4(worldFromPair);
        camera.up.transformDirection(worldFromPair);
        target.applyMatrix4(worldFromPair);
      }
      camera.lookAt(target);
    }
    draw();
  }
  function rememberView() {
    parent.postMessage(
      {
        studioViewport: 'camera',
        camera: {
          name: select.selectedOptions[0].text,
          position: camera.position.toArray(),
          target: controls.target.toArray(),
        },
      },
      '*',
    );
  }
  controls.addEventListener('end', rememberView);
  select.onchange = () => {
    setView();
    rememberView();
  };
  const resize = () => {
    const { clientWidth: w, clientHeight: h } = document.documentElement;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    draw();
  };
  const previous = location.hash ? JSON.parse(decodeURIComponent(location.hash.slice(1))) : null;
  new ResizeObserver(resize).observe(document.documentElement);
  return (nextScene, nextJob, far) => {
    const first = !scene;
    const name = first ? previous?.name : select.selectedOptions[0]?.text;
    scene = nextScene;
    job = nextJob;
    select.length = 1;
    job.streams.forEach((stream, i) => select.add(new Option(stream, String(i))));
    const option = [...select.options].find((item) => item.text === name);
    select.value = option?.value ?? 'orbit';
    camera.far = Math.max(12, far);
    camera.updateProjectionMatrix();
    renderer.shadowMap.needsUpdate = true;
    if (first || !option || select.value !== 'orbit') setView();
    if (first && option && controls.enabled && previous) {
      camera.position.fromArray(previous.position);
      controls.target.fromArray(previous.target);
      controls.update();
    }
    resize();
  };
}
