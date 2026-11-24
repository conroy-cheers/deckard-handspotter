import * as THREE from 'three';
import { OrbitControls } from 'orbit-controls';

const get = id => document.getElementById(id);
const vec = xyz => new THREE.Vector3(...xyz);
const radians = Math.PI / 180;
const colors = [0x63cae7,0x76e1a3,0xffbf68,0xe39bf6];

function arrayFromBase64(value, type) {
  return new type(Uint8Array.fromBase64(value).buffer);
}

function line(points, color, loop=false) {
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({color, transparent:loop, opacity:loop ? .42 : 1});
  return loop ? new THREE.LineLoop(geo,material) : new THREE.Line(geo,material);
}

function markerSphere(radius, color) {
  return new THREE.Mesh(new THREE.SphereGeometry(radius,16,12),new THREE.MeshBasicMaterial({color}));
}

function start() {
  const data = JSON.parse(get('data').textContent);
  const viewport = get('viewport');
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  viewport.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const view = new THREE.PerspectiveCamera(40,1,.012,4);
  const controls = new OrbitControls(view,renderer.domElement);
  controls.minDistance=.22;
  controls.maxDistance=2.2;
  scene.add(new THREE.HemisphereLight(0xe6f2ff,0x66717d,2.1));
  const light = new THREE.DirectionalLight(0xffece3,2.4);
  light.position.set(.1,.65,-.55);
  scene.add(light);
  const grid = new THREE.GridHelper(1.4,14,0x435366,0x293a4b);
  grid.position.y=-.50;
  scene.add(grid);

  const rig = new THREE.Group();
  const armGroup = new THREE.Group();
  const eye = markerSphere(.007,0xf2f5ff);
  eye.position.copy(vec(data.eyes));
  rig.add(eye);
  const pivot = markerSphere(.009,0xf5b6a9);
  scene.add(rig,armGroup,pivot);

  const torsoMaterial = new THREE.MeshStandardMaterial({color:0x9aaec4,roughness:.76,side:THREE.DoubleSide});
  const armMaterial = torsoMaterial.clone();
  const head = new THREE.Mesh(new THREE.BufferGeometry(),[torsoMaterial,armMaterial]);
  scene.add(head);
  const labels = [];
  function addLabel(object, text, color) {
    const element = document.createElement('span');
    element.className='marker';
    element.textContent=text;
    element.style.color=new THREE.Color(color).getStyle();
    get('labels').append(element);
    labels.push({object,element});
  }
  addLabel(eye,'Model eye midpoint',0xf2f5ff);
  addLabel(pivot,'Neck pivot',0xf5b6a9);

  const sensors = data.cameras.map((camera,i)=>{
    const color=colors[i];
    const node = new THREE.Group();
    const housing = new THREE.Mesh(new THREE.BoxGeometry(.024,.016,.009),new THREE.MeshStandardMaterial({color}));
    housing.position.z=-.0045;
    node.add(housing,markerSphere(.004,color));
    node.add(new THREE.ArrowHelper(vec([0,0,1]),vec([0,0,0]),.21,color,.025,.012));
    const support=new THREE.Group();
    const boundary=camera.rim.map(point=>vec(point).multiplyScalar(.26));
    support.add(line(boundary,color,true));
    for (let i=0;i<boundary.length;i+=Math.max(1,Math.floor(boundary.length/8))) {
      support.add(line([vec([0,0,0]),boundary[i]],color,true));
    }
    node.add(support);
    addLabel(node,camera.sensor.replaceAll('_',' '),color);
    rig.add(node);
    return {camera,node,support};
  });

  let current, neutral, weights;
  const angles=['yaw','pitch','roll'].map(get);
  const root = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const axis = new THREE.Vector3();
  const partial = new THREE.Quaternion();

  function draw() {
    renderer.render(scene,view);
    for (const {object,element} of labels) {
      position.setFromMatrixPosition(object.matrixWorld).project(view);
      element.hidden=!get('markers').checked || position.z>1 || position.z< -1 ||
        Math.abs(position.x)>1 || Math.abs(position.y)>1;
      if (!element.hidden) {
        element.style.left=`${(position.x+1)*viewport.clientWidth/2}px`;
        element.style.top=`${(1-position.y)*viewport.clientHeight/2}px`;
      }
    }
  }

  function updatePose() {
    const [yaw,pitch,roll]=angles.map(el=>{
      getOutput(el).value=`${Number(el.value).toFixed(1)}°`;
      return Number(el.value)*radians;
    });
    rotation.setFromEuler(new THREE.Euler(pitch,-yaw,roll,'YXZ'));
    rig.quaternion.copy(rotation);
    rig.position.copy(pivot.position).sub(position.copy(pivot.position).applyQuaternion(rotation));
    const theta=2*Math.acos(THREE.MathUtils.clamp(rotation.w,-1,1));
    const qnorm=Math.hypot(rotation.x,rotation.y,rotation.z);
    const target=head.geometry.attributes.position;
    target.array.set(neutral);
    if (qnorm>1.e-10) {
      axis.set(rotation.x/qnorm,rotation.y/qnorm,rotation.z/qnorm);
      for (let i=0;i<weights.length;i++) {
        if (!weights[i]) continue;
        partial.setFromAxisAngle(axis,theta*weights[i]);
        position.fromArray(neutral,3*i).sub(pivot.position).applyQuaternion(partial).add(pivot.position);
        target.setXYZ(i,position.x,position.y,position.z);
      }
    }
    target.needsUpdate=true;
    head.geometry.computeVertexNormals();
    head.geometry.computeBoundingSphere();
    draw();
  }

  function getOutput(input) {
    return input.parentElement.querySelector('output');
  }

  function setAngles(values) {
    angles.forEach((item,i)=>{item.value=values[i];});
    updatePose();
  }

  function selectWeights() {
    const field=get('weights').value;
    weights=arrayFromBase64(current[field],Float32Array);
    const armCount=current.arm_vertices_with_head_weight;
    get('deformation').textContent=field==='neutral_weights'
      ? 'Comparison uses connected head and neck vertices from the neutral body.'
      : `${armCount.toLocaleString()} arm vertices receive more than half the head rotation in this scene.`;
    updatePose();
  }

  function setBody() {
    current=data.bodies[Number(get('body').value)];
    neutral=arrayFromBase64(current.positions,Float32Array);
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(neutral.slice(),3).setUsage(THREE.DynamicDrawUsage));
    geo.setIndex(new THREE.BufferAttribute(arrayFromBase64(current.indices,Uint32Array),1));
    geo.addGroup(0,current.torso_indices,0);
    geo.addGroup(current.torso_indices,geo.index.count-current.torso_indices,1);
    head.geometry.dispose();
    head.geometry=geo;
    pivot.position.copy(vec(current.pivot));
    for (const object of [...armGroup.children]) {
      armGroup.remove(object);
      object.geometry.dispose();
      object.material.dispose();
    }
    for (const [side,arm] of Object.entries(current.arms)) {
      const color=side==='left' ? colors[0] : colors[3];
      const chain=[arm.shoulder,arm.elbow,arm.wrist].map(vec);
      armGroup.add(line(chain,color));
      for (const joint of chain) {
        const dot=markerSphere(.007,color);
        dot.position.copy(joint);
        armGroup.add(dot);
      }
      for (let first=1;first<21;first+=4) {
        armGroup.add(line([arm.fingers[0],...arm.fingers.slice(first,first+4)].map(vec),color));
      }
    }
    angles.forEach(input=>{input.value=0;});
    selectWeights();
  }

  function setMount() {
    const mode=get('mount').value;
    get('placement').textContent=mode==='renderer'
      ? 'Front-camera midpoint sits 95 mm ahead of the modeled eyes. The renderer levels the mean front optical axis.'
      : 'Measured OpenVR camera orientations, translated to the same front-camera midpoint. Fit to the face remains an assumption.';
    const rows=sensors.map(({camera,node},i)=>{
      const matrix=camera[mode];
      root.set(...matrix.flat());
      root.decompose(node.position,node.quaternion,node.scale);
      const row=document.createElement('tr');
      const name=row.insertCell();
      name.textContent=camera.sensor;
      name.style.color=new THREE.Color(colors[i]).getStyle();
      name.title=`Measured OpenVR origin (mm)  ${camera.factory_mm.join(', ')}`;
      const center=data.eyes.map((eye,j)=>Math.round((matrix[j][3]-eye)*1000));
      const aim=mode==='renderer' ? camera.aim : camera.openvr_aim;
      for (const value of [...center,...aim.map(angle=>`${angle.toFixed(1)}°`)]) row.insertCell().textContent=value;
      return row;
    });
    get('cameras').replaceChildren(...rows);
    draw();
  }

  function overlays() {
    for (const {support} of sensors) support.visible=get('rims').checked;
    armGroup.visible=get('arms').checked;
    armMaterial.visible=get('skin').checked;
    for (const material of head.material) {
      material.transparent=get('ghost').checked;
      material.opacity=material.transparent ? .25 : 1;
      material.depthWrite=!material.transparent;
      material.needsUpdate=true;
    }
    draw();
  }

  function look(from) {
    controls.target.set(0,-.095,.07).applyQuaternion(rig.quaternion).add(rig.position);
    view.position.copy(controls.target).add(vec(from).applyQuaternion(rig.quaternion));
    controls.update();
    draw();
  }
  const presets={corner:[.51,.21,-.78],profile:[.85,.02,.01],front:[0,.04,-.9],top:[0,.91,-.006]};
  for (const [id,point] of Object.entries(presets)) get(id).addEventListener('click',()=>look(point));
  controls.addEventListener('change',draw);
  const resize = ()=>{
    const {clientWidth:width, clientHeight:height}=viewport;
    renderer.setSize(width,height,false);
    view.aspect=width/height;
    view.updateProjectionMatrix();
    draw();
  };
  new ResizeObserver(resize).observe(viewport);
  for (const [key,body] of data.bodies.entries()) {
    get('body').add(new Option(`Scene ${body.id}`,String(key)));
  }
  get('body').addEventListener('change',setBody);
  get('mount').addEventListener('change',setMount);
  get('weights').addEventListener('change',selectWeights);
  angles.forEach(input=>input.addEventListener('input',updatePose));
  get('neutral').addEventListener('click',()=>setAngles([0,0,0]));
  get('recorded').addEventListener('click',()=>setAngles(current.angles));
  for (const key of ['rims','arms','skin','ghost','markers']) get(key).addEventListener('change',overlays);
  setBody();
  setMount();
  overlays();
  look(presets.corner);
  resize();
  document.documentElement.dataset.ready='true';
}

try {
  start();
} catch (error) {
  get('error').textContent=`Could not load the camera viewer. ${error.message}`;
  get('error').hidden=false;
  throw error;
}
