import { sensorCapture } from '/pbr_sensor.mjs';
import * as THREE from 'three';
import { buildScene, trimViewportRooms, loadMaterials, skinMaterial, torsoMaterial } from '/pbr_scene.mjs';
import { bindArmVelocities, cameraMotion, motionMaterial } from '/pbr_motion.mjs';

async function fetchInput(path) {
  const response=await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}: ${response.status}`);
  return response;
}

async function loadSceneInput(frame,binarySurfaces,base='') {
  if (!binarySurfaces) return (await fetchInput(base+'/'+frame+'.json')).json();
  const buffer=await (await fetchInput(base+'/'+frame+'.bin')).arrayBuffer();
  const header=new DataView(buffer).getUint32(0,true);
  const start=4+header;
  const data=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,header)));
  const types={'<f4':Float32Array,'<u4':Uint32Array};
  const surfaces=data.skin_contact ? [...data.surfaces,data.skin_contact] : data.surfaces;
  for (const surface of surfaces) {
    for (const field of ['position','normal','faces']) {
      if (surface[field]===undefined) continue;
      const {dtype,offset,shape}=data.arrays[surface[field]];
      surface[field]=new types[dtype](buffer,start+offset,shape[0]);
    }
  }
  return data;
}

function partMaterial(id) {
  const material = new THREE.ShaderMaterial({
    uniforms: {part: {value: id/255}},
    side: id===0 ? THREE.FrontSide : THREE.DoubleSide,
    vertexShader: 'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader: 'uniform float part; void main(){gl_FragColor=vec4(part,0.,0.,1.);}',
  });
  material.userData.capturePart=id;
  return material;
}

function makeMesh(description, materials, parts) {
  const geo=new THREE.BufferGeometry();
  for (const [attribute, key] of [
    ['position','position'], ['normal','normal'], ['rest','rest'], ['restNormal','rest_normal'], ['regions','regions'],
  ]) geo.setAttribute(attribute, new THREE.Float32BufferAttribute(description[key],3));
  geo.setIndex(description.faces);
  const handIndices=description.hand_triangles*3;
  geo.addGroup(0,handIndices,0);
  geo.addGroup(handIndices,description.faces.length-handIndices,1);
  const skin=skinMaterial(materials,description.skin,description.cuff,description.palm_lines,description.appearance);
  const mesh=new THREE.Mesh(geo,[skin,skin]);
  mesh.userData.partMaterial=[parts[description.part],parts[3]];
  mesh.castShadow=mesh.receiveShadow=true;
  return mesh;
}

function makeTorso(description, materials, part) {
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(description.position,3));
  geometry.setAttribute('normal',new THREE.Float32BufferAttribute(description.normal,3));
  geometry.setIndex(description.faces);
  const mesh=new THREE.Mesh(geometry,torsoMaterial(materials,description.appearance));
  mesh.userData.partMaterial=part;
  mesh.castShadow=mesh.receiveShadow=true;
  return mesh;
}

function makeBakedSurface(description, appearance, materials, part) {
  let geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(description.position,3));
  geo.setIndex(Array.isArray(description.faces) ? description.faces :
    new THREE.BufferAttribute(description.faces,1));
  if(description.normal) geo.setAttribute('normal',new THREE.Float32BufferAttribute(description.normal,3));
  else geo.computeVertexNormals();
  if(description.panel_uv){
    // Sewn vertices share simulation indices, while render corners keep
    // the panel-local coordinates on their own side of each UV seam.
    const original=geo;
    geo=original.toNonIndexed();
    original.dispose();
    geo.setAttribute('uv',new THREE.Float32BufferAttribute(
      description.uv_faces.flat().flatMap(index=>description.panel_uv[index]),2));
  }
  const material=description.material==='cloth' ? torsoMaterial(materials,appearance) :
    new THREE.MeshStandardMaterial({color:appearance?.skin_color ?? 0xB57D62,roughness:.62});
  material.side=THREE.DoubleSide;
  const mesh=new THREE.Mesh(geo,material);
  mesh.userData.partMaterial=part;
  mesh.castShadow=mesh.receiveShadow=true;
  return mesh;
}

const foundationMasks=new WeakMap();
function foundationMask(material) {
  if (!foundationMasks.has(material)) {
    const mask=new THREE.MeshBasicMaterial({
      color: 0, map: material.map, alphaMap: material.alphaMap, alphaTest: material.alphaTest,
      side: material.side, toneMapped: false,
    });
    mask.visible=material.visible && !(material.transmission>=.85);
    foundationMasks.set(material,mask);
  }
  return foundationMasks.get(material);
}

async function renderJob(report) {
  const version=Number(new URLSearchParams(location.search).get('version'));
  const base=version ? '/v/'+version : '';
  const job=await (await fetchInput(base+'/job.json')).json();
  const {size}=job;
  const canvas=document.createElement('canvas');
  const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
  if (!gl) throw new Error('WebGL 2 is unavailable');
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('Float color targets are unavailable');
  const debug=gl.getExtension('WEBGL_debug_renderer_info');
  report.device=gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
  const renderer=new THREE.WebGLRenderer({canvas,context:gl});
  renderer.debug.onShaderError=(context,program,vert,frag)=>{
    throw new Error([context.getProgramInfoLog(program),context.getShaderInfoLog(vert),
        context.getShaderInfoLog(frag)].join('\n'));
  };
  renderer.setSize(size,size);
  renderer.toneMapping=THREE.NoToneMapping;
  renderer.shadowMap.type=THREE.PCFShadowMap;
  renderer.shadowMap.autoUpdate=false;
  const parts=[0,1,2,3].map(partMaterial);
  const debrisGeometry=new THREE.DodecahedronGeometry(1,0);
  const debrisMaterials=[0x554538,0xb5a593].map(color=>new THREE.MeshStandardMaterial({color,roughness:.9}));
  const materials=await loadMaterials(renderer);
  const defaultFar=8;
  async function buildSubject(input,wardrobe) {
    const {meshes,torso,surfaces=[],appearance,debris=[],garment,skin_contact}=input;
    if (garment){
      const {surface,statistics}=await wardrobe.dress(garment,skin_contact ? [skin_contact] : surfaces);
      if (garment.panel_uv) Object.assign(surface, {panel_uv:garment.panel_uv, uv_faces:garment.uv_faces});
      surfaces.push(surface);
      report.wardrobe_cases?.push(statistics);
    }
    const hands=meshes.map(description=>makeMesh(description,materials,parts));
    const bodyParts=surfaces.map(s=>makeBakedSurface(s,appearance,materials,parts[s.part]));
    if (torso) bodyParts.push(makeTorso(torso,materials,parts[3]));
    const bodyMaterials=bodyParts.map(mesh=>mesh.material);
    const fragments=[];
    for (const piece of debris) {
      const fragment=new THREE.Mesh(debrisGeometry,debrisMaterials[piece.material]);
      fragment.position.fromArray(piece.position);
      fragment.scale.fromArray(piece.scale);
      fragment.rotation.fromArray(piece.rotation);
      fragments.push(fragment);
    }
    return {hands,bodyParts,bodyMaterials,meshes,surfaces,objects:[...hands,...bodyParts,...fragments]};
  }
  async function makeRoom(job,layout,lighting,base) {
    const scene=await buildScene(materials,layout,lighting,base,JSON.stringify(job.foundation_key));
    return scene;
  }
  function applyCaptureTransform(scene,toPair) {
    if (toPair) {
      // Express the fixed room, body and lights in the turned headset frame.
      // All cameras keep their calibrated positions and axes in that frame.
      scene.matrix.set(...toPair.flat());
      scene.matrixAutoUpdate=false;
      scene.matrixWorldNeedsUpdate=true;
      const turn=new THREE.Quaternion().setFromRotationMatrix(scene.matrix);
      const environment=new THREE.Quaternion().setFromEuler(scene.environmentRotation);
      scene.environmentRotation.setFromQuaternion(turn.multiply(environment));
    }
  }

  function prepareCapture(job) {
    const {size}=job;
    const beautyTarget=new THREE.WebGLArrayRenderTarget(size,size,5,{
      type:THREE.FloatType,colorSpace:THREE.LinearSRGBColorSpace,
    });
    const labelTarget=new THREE.WebGLArrayRenderTarget(size,size,5,{type:THREE.FloatType});
    const sensorPass=sensorCapture(renderer);
    const cameras=Array.from({length:5},()=>new THREE.PerspectiveCamera(90,1,.02,defaultFar));
    async function draw(job,frame,subject,scene,layout,lightingExposure,base) {
      const {hands,bodyParts,meshes,surfaces}=subject;
      const motion=job.motion?.[frame];
      const useFlow=!!(motion?.arms || motion?.camera);
      const sensor=useFlow ? cameraMotion(motion.camera,job.rig_center) : null;
      if (motion?.arms) {
        for (const [index,mesh] of hands.entries()) {
          const side=meshes[index].part===1 ? 'left' : 'right';
          if (motion.arms[side]) bindArmVelocities(mesh,{[side]:motion.arms[side]});
        }
        // Baked arms and sleeves share the same velocity field so vertices at
        // the shoulder and at the hand's semantic boundary remain continuous.
        for (const mesh of bodyParts.slice(0,surfaces.length)) bindArmVelocities(mesh,motion.arms);
      }
      const draws=[], staticMasks=new Map(), movingMasks=new Map();
      scene.traverse(object=>{
        if (!object.isMesh) return;
        const { material }=object;
        let label=object.userData.partMaterial||parts[0];
        if (object.userData.foundationMesh) {
          label=Array.isArray(material) ? material.map(foundationMask) : foundationMask(material);
        }
        if (useFlow) {
          const moving=!!object.geometry.attributes.armVelocity;
          const cache=moving ? movingMasks : staticMasks;
          const withMotion=mask=>{
            if (!cache.has(mask)) cache.set(mask,motionMaterial(mask,moving,sensor));
            return cache.get(mask);
          };
          label=Array.isArray(label) ? label.map(withMotion) : withMotion(label);
        }
        draws.push([object,material,label]);
      });
      const background=scene.background;
      const far=layout.foundation?.far ?? defaultFar;
      renderer.shadowMap.needsUpdate=true;
      try {
        for (const [eye,faces] of job.eyes.entries()){
          for (const [face,axes] of faces.entries()){
            const camera=cameras[face];
            camera.position.fromArray(axes.position);
            camera.up.fromArray(axes.up);
            camera.lookAt(...axes.look);
            if (camera.far!==far) {
              camera.far=far;
              camera.updateProjectionMatrix();
            }
            for (const [object,beauty] of draws) object.material=beauty;
            scene.background=background;
            renderer.shadowMap.enabled=true;
            renderer.setRenderTarget(beautyTarget,face);
            renderer.render(scene,camera);
            for (const [object,,segmentation] of draws) object.material=segmentation;
            scene.background=null;
            renderer.setClearColor(0,useFlow ? 0 : 1);
            renderer.shadowMap.enabled=false;
            renderer.setRenderTarget(labelTarget,face);
            renderer.render(scene,camera);
          }
          const output=sensorPass.read(job,frame,eye,lightingExposure,beautyTarget.texture,labelTarget.texture,
            useFlow ? motion : null,sensor?.angular ?? new THREE.Vector3(),report.optics_validation);
          const response=await fetch(`${base}/output/${frame}-${eye}.bin`,{method:'POST',body:output,headers:job.capture ? {'x-capture-token':job.capture.token} : {}});
          if (!response.ok) throw new Error(`Surface readback failed for view ${frame}-${eye}`);
        }
      } finally {
        for (const [object,beauty] of draws) object.material=beauty;
        scene.background=background;
        renderer.setRenderTarget(null);
        renderer.shadowMap.enabled=true;
        renderer.shadowMap.needsUpdate=true;
        for (const material of staticMasks.values()) material.dispose();
        for (const material of movingMasks.values()) material.dispose();
      }
    }
    return {draw,dispose(){sensorPass.dispose();beautyTarget.dispose();labelTarget.dispose();}};
  }

  if (job.viewport != null) {
    const {showViewport} = await import('/viewport.mjs');
    const view=showViewport(renderer);
    let subject, activeScene, geometry, roomKey, capture, captureSize;
    const sharedMaterials=new Set([...Object.values(materials),...debrisMaterials]);
    function disposeObjects(root) {
      root.traverse(object=>{
        object.shadow?.dispose();
        if (!object.isMesh || object.userData.foundationMesh) return;
        if (object.geometry !== debrisGeometry) object.geometry.dispose();
        for (const material of new Set([object.material].flat())) {
          if (!sharedMaterials.has(material)) material.dispose();
        }
      });
    }
    async function update(job,base) {
      const reuse=subject && geometry===job.geometry;
      let next=subject;
      if (!reuse) {
        const input=await loadSceneInput(job.sample,job.binary_surfaces,job.geometry ? '/v/'+job.geometry : base);
        const wardrobe=job.wardrobe_mode ?
          await (await import('/pbr_cloth.mjs')).wardrobeRenderer() : null;
        try { next=await buildSubject(input,wardrobe); }
        finally { wardrobe?.dispose(); }
        next.room={layout:input.layout,lighting:input.lighting};
      }
      const {layout,lighting}=job.room || next.room;
      const nextRoomKey=JSON.stringify([layout,lighting,job.foundation_key]);
      if (!reuse || nextRoomKey!==roomKey) {
        let scene;
        try { scene=await makeRoom(job,layout,lighting,base); }
        catch(error) {
          if (!reuse) next.objects.forEach(disposeObjects);
          throw error;
        }
        for (const object of next.objects) scene.add(object);
        if (activeScene) disposeObjects(activeScene);
        activeScene=scene;
        geometry=job.geometry;
        subject=next;
        roomKey=nextRoomKey;
        await trimViewportRooms();
      }
      if (job.capture) {
        if (captureSize!==job.size) {
          capture?.dispose();
          captureSize=null;
          capture=prepareCapture(job);
          captureSize=job.size;
        }
        const environment=activeScene.environmentRotation.clone();
        try {
          applyCaptureTransform(activeScene,job.pair_from_reference?.[0]);
          await capture.draw(job,0,subject,activeScene,layout,lighting.exposure,base);
        } finally {
          activeScene.matrix.identity();
          activeScene.matrixAutoUpdate=true;
          activeScene.matrixWorldNeedsUpdate=true;
          activeScene.environmentRotation.copy(environment);
        }
      }
      view(activeScene,job,layout.foundation?.far ?? defaultFar);
    }
    await update(job,base);
    let queue=Promise.resolve(),latest;
    window.addEventListener('message',event=>{
      if(event.source!==parent || !Number.isSafeInteger(event.data?.version)) return;
      const version=latest=event.data.version;
      queue=queue.then(async()=>{
        if(version!==latest) return;
        const base='/v/'+version;
        const job=await (await fetchInput(base+'/job.json')).json();
        await update(job,base);
        parent.postMessage({studioViewport:'ready',version,setup:job.viewport_setup},'*');
      }).catch(error=>parent.postMessage({studioViewport:'error',version,message:error.message},'*'));
    });
    parent.postMessage({studioViewport:'ready',version,setup:job.viewport_setup},'*');
    return;
  }
  report.optics_validation=[];
  const capture=prepareCapture(job);
  const wardrobe=job.wardrobe_mode ?
    await (await import('/pbr_cloth.mjs')).wardrobeRenderer() : null;
  if(wardrobe) report.wardrobe_cases=[];
  const start=performance.now();
  for (let frame=0;frame<job.count;frame++){
    const input=await loadSceneInput(job.sample ?? frame,job.binary_surfaces);
    const {layout,lighting}=input;
    const subject=await buildSubject(input,wardrobe);
    const {hands,bodyMaterials,objects}=subject;
    const scene=await makeRoom(job,layout,lighting,'');
    for (const object of objects) scene.add(object);
    applyCaptureTransform(scene,job.pair_from_reference?.[frame]);
    await capture.draw(job,frame,subject,scene,layout,lighting.exposure,'');
    scene.traverse(object=>{
      if (object.isMesh && !object.userData.foundationMesh && object.geometry!==debrisGeometry) object.geometry.dispose();
    });
    for (const hand of hands) hand.material[0].dispose();
    for (const material of bodyMaterials) material.dispose();
    scene.traverse(object=>object.shadow?.map?.dispose());
  }
  capture.dispose();
  debrisGeometry.dispose();
  wardrobe?.dispose();
  for (const material of debrisMaterials) material.dispose();
  report.pairs=job.count;
  report.photometry='linear-srgb-luminance';
  report.sensor_noise='pcg-box-muller-v1';
  report.render_seconds=(performance.now()-start)/1000;
}

const report={three:THREE.REVISION,navigator:navigator.userAgent};
try {
  await renderJob(report);
} catch(error) {
  report.error=error.stack||String(error);
}
if (window.parent !== window && report.error) parent.postMessage({studioViewport:'error',message:report.error}, '*');
else if (window.parent === window) await fetch('/result',{method:'POST',body:JSON.stringify(report)});
