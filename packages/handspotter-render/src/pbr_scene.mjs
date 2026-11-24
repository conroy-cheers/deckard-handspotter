import * as THREE from 'three';
import { HDRLoader } from '/three/examples/jsm/loaders/HDRLoader.js';
import { RectAreaLightUniformsLib } from '/three/examples/jsm/lights/RectAreaLightUniformsLib.js';

const textureLoader = new THREE.TextureLoader();
const foundationModels = new Map();

async function texture(name, color = false) {
  const map = await textureLoader.loadAsync('/assets/'+name);
  map.colorSpace = color ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.anisotropy = 4;
  return map;
}

function standard(color, roughness, metalness = 0) {
  return new THREE.MeshStandardMaterial({color, roughness, metalness});
}

function paintedWall() {
  const paint=standard(0x8f8c84,.92);
  paint.onBeforeCompile=shader=>{
    shader.vertexShader='varying vec3 paintPosition;\n'+shader.vertexShader.replace(
      '#include <begin_vertex>', '#include <begin_vertex>\npaintPosition=position;');
    shader.fragmentShader='varying vec3 paintPosition;\n'+shader.fragmentShader.replace(
      '#include <color_fragment>', `
      #include <color_fragment>
      vec3 p=paintPosition;
      float broad=sin(p.x*13.8+p.y*5.6+p.z*7.1)*sin(p.y*12.3-p.z*8.3);
      float fine=sin(p.x*77.+p.y*61.)*sin(p.z*71.-p.y*59.);
      diffuseColor.rgb*=.94+.048*broad+.023*fine;
    `);
  };
  return paint;
}

export async function loadMaterials(renderer) {
  RectAreaLightUniformsLib.init();
  const pmrem = new THREE.PMREMGenerator(renderer);
  const indoors = {};
  for (const name of ['desk', 'workbench']) {
    const hdr = await new HDRLoader().loadAsync('/assets/'+name+'.hdr');
    indoors[name] = pmrem.fromEquirectangular(hdr);
    hdr.dispose();
  }
  pmrem.dispose();
  async function surface(name, bumpScale) {
    const [map, normalMap, roughnessMap] = await Promise.all([
      texture(name+'-color.jpg', true), texture(name+'-normal.jpg'), texture(name+'-roughness.jpg'),
    ]);
    return new THREE.MeshStandardMaterial({
      map, normalMap, roughnessMap, normalScale: new THREE.Vector2(bumpScale,bumpScale), roughness: 1,
    });
  }
  const [wood, concrete, fabric, skinColor, skinRoughness, skinHeight] = await Promise.all([
    surface('wood', .38), surface('concrete', .24), surface('fabric', .55),
    texture('skin-color.jpg', true), texture('skin-roughness.jpg'), texture('skin-height.png'),
  ]);
  const floor=concrete.clone();
  floor.color.setRGB(.47,.46,.44);
  fabric.userData.tileMeters=.39;
  return {indoors, wood, concrete, floor, fabric, skinColor, skinRoughness, skinHeight,
    ceramic: standard(0xe1dfd7,.26), steel: standard(0xa7a6a3,.44,.85),
    plastic: standard(0x252a2b,.63), wall: paintedWall(),
    paper: standard(0xe0d9c5,.91), enamel: standard(0x979da0,.52),
    ink: standard(0x393732,.94), tile: standard(0xc5c7c1,.48),
    window: new THREE.MeshStandardMaterial({color:0x8fa2ad,emissive:0xa4c5d4,emissiveIntensity:.62,roughness:.67}),
  };
}

const fabricShader = `
  vec4 sampleSurface(sampler2D map, float frequency, vec3 position, vec3 normal) {
    vec3 p=position*frequency;
    vec3 w=pow(abs(normalize(normal)),vec3(5.));
    w/=max(dot(w,vec3(1.)),0.00001);
    return texture2D(map,p.yz)*w.x+texture2D(map,p.xz)*w.y+texture2D(map,p.xy)*w.z;
  }
  float knitStripe(float phase) {
    return .5+.5*cos(phase)*exp(-.18*pow(fwidth(phase),2.));
  }
  float fabricShade(sampler2D map, vec3 position, vec3 normal, float style, float across) {
    vec3 fibers=sampleSurface(map,12.,position,normal).rgb;
    float yarn=.59+.64*dot(fibers,vec3(.2126,.7152,.0722));
    if (style==1.) {
      yarn*=.62+.40*knitStripe(position.y*177.)+.23*knitStripe(across*8.);
    } else if (style==2.) {
      yarn*=.72+.35*knitStripe(across*43.);
    } else if (style==3.) {
      yarn*=.53+.69*pow(knitStripe(position.y*167.),5.);
    }
    return yarn;
  }
`;

function garmentColor(appearance) {
  return appearance ? new THREE.Color(appearance.cloth_color) : new THREE.Color(.040,.048,.056);
}

export function torsoMaterial(materials, appearance) {
  const material=new THREE.MeshStandardMaterial({color:garmentColor(appearance),
    roughness:appearance?.cloth_roughness ?? .92});
  material.onBeforeCompile=shader=>{
    shader.uniforms.fabricMap={value:materials.fabric.map};
    shader.uniforms.pattern={value:appearance?.pattern ?? 0};
    const attributes='varying vec3 garmentPosition, garmentNormal;\n';
    shader.vertexShader=attributes+shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      garmentPosition=position;
      garmentNormal=normal;
    `);
    shader.fragmentShader=attributes+'uniform sampler2D fabricMap; uniform float pattern;\n'+fabricShader+
      shader.fragmentShader.replace('#include <color_fragment>', `
        #include <color_fragment>
        diffuseColor.rgb*=fabricShade(fabricMap,garmentPosition,garmentNormal,pattern,18.*garmentPosition.x);
      `);
  };
  return material;
}

export function skinMaterial(materials, tone, cuff, creases, appearance) {
  const color = appearance ? new THREE.Color(appearance.skin_color)
    : new THREE.Color(0x67412a).lerp(new THREE.Color(0xd6ae92),tone);
  const material = new THREE.MeshPhysicalMaterial({
    color, roughness: .68, metalness: 0, ior: 1.43, specularIntensity: .54, side: THREE.DoubleSide,
  });
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, {
      skinColor: {value: materials.skinColor}, skinRough: {value: materials.skinRoughness},
      skinHeight: {value: materials.skinHeight}, cuffHeight: {value: cuff},
      sleeveMap: {value: materials.fabric.map},
      clothColor: {value: garmentColor(appearance)},
      clothPattern: {value: appearance?.pattern ?? 0},
      clothRoughness: {value: appearance?.cloth_roughness ?? .92},
      palmLines: {value: creases.map(segment=>new THREE.Vector4(...segment))},
    });
    shader.vertexShader = `
      attribute vec3 rest;
      attribute vec3 restNormal;
      attribute vec3 regions;
      varying vec3 bindPosition, bindNormal, anatomy;
    ` + shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      bindPosition=rest; bindNormal=restNormal; anatomy=regions;
    `);
    shader.fragmentShader = `
      uniform sampler2D skinColor, skinRough, skinHeight, sleeveMap;
      uniform float cuffHeight, clothPattern, clothRoughness;
      uniform vec3 clothColor;
      uniform vec4 palmLines[9];
      varying vec3 bindPosition, bindNormal, anatomy;
      ${fabricShader}
      vec4 skinSample(sampler2D map, float frequency) {
        return sampleSurface(map,frequency,bindPosition,bindNormal);
      }
      float palmLine() {
        float line=0.;
        for (int i=0;i<9;i++) {
          vec2 start=palmLines[i].xy;
          vec2 path=palmLines[i].zw-start;
          vec2 p=bindPosition.xy-start;
          float t=clamp(dot(p,path)/max(dot(path,path),1e-8),0.,1.);
          line=max(line,1.-smoothstep(.00040,.00153,length(p-t*path)));
        }
        return line;
      }
    ` + shader.fragmentShader
      .replace('#include <color_fragment>', `
        #include <color_fragment>
        float sleeve=1.-smoothstep(cuffHeight-.002,cuffHeight+.002,bindPosition.y);
        float nail=smoothstep(.45,.72,anatomy.z);
        float palm=anatomy.x*(1.-nail);
        float knuckle=anatomy.y*(1.-nail);
        float crease=palm*palmLine()*(1.-sleeve);
        vec3 tile=skinSample(skinColor,44.).rgb;
        vec3 pos=bindPosition;
        float mottle=sin(pos.x*113.+sin(pos.z*168.))*sin(pos.y*103.-pos.z*94.);
        diffuseColor.rgb*=(vec3(.79)+tile*.74)*(1.+mottle*.075);
        diffuseColor.rgb*=mix(vec3(1.),vec3(1.21,1.14,1.11),palm);
        diffuseColor.rgb*=mix(vec3(1.),vec3(.88,.77,.74),knuckle*.67);
        diffuseColor.rgb*=1.-.17*crease;
        vec3 nailColor=mix(diffuseColor.rgb,vec3(.63,.47,.42),.36);
        diffuseColor.rgb=mix(diffuseColor.rgb,nailColor,nail);
        float yarn=fabricShade(sleeveMap,pos,bindNormal,clothPattern,atan(pos.z,pos.x));
        diffuseColor.rgb=mix(diffuseColor.rgb,clothColor*yarn,sleeve);
      `)
      .replace('#include <roughnessmap_fragment>', `
        #include <roughnessmap_fragment>
        roughnessFactor=mix(.61+.31*skinSample(skinRough,44.).g,.29,nail);
        roughnessFactor=mix(roughnessFactor,clothRoughness,sleeve);
      `)
      .replace('#include <normal_fragment_maps>', `
        #include <normal_fragment_maps>
        float h=(skinSample(skinHeight,57.).r-.5)*.00030*(1.-nail)*(1.-sleeve)-.00023*crease;
        vec3 sx=dFdx(-vViewPosition), sy=dFdy(-vViewPosition);
        vec3 r1=cross(sy,normal), r2=cross(normal,sx);
        float det=dot(sx,r1)*faceDirection;
        normal=normalize(max(abs(det),1e-10)*normal-sign(det)*(dFdx(h)*r1+dFdy(h)*r2));
      `);
  };
  return material;
}

function put(scene, geometry, material, x, y, z) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x,y,z);
  mesh.castShadow = mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
}

function box(scene, material, w,h,d, x,y,z) {
  const geometry=new THREE.BoxGeometry(w,h,d);
  if (material.map) {
    const uv=geometry.attributes.uv;
    // BoxGeometry emits four vertices for each of the X, Y, and Z faces.
    const dimensions=[[d,h],[d,h],[w,d],[w,d],[w,h],[w,h]];
    const scale=material.userData.tileMeters ?? .78;
    for (const [face,[width,height]] of dimensions.entries()) {
      for (let vertex=4*face;vertex<4*(face+1);vertex++) {
        uv.setXY(vertex,uv.getX(vertex)*width/scale,uv.getY(vertex)*height/scale);
      }
    }
  }
  return put(scene,geometry,material,x,y,z);
}

function cup(scene, material, x,y,z, scale = 1) {
  const profile = [[0,0],[.026,0],[.034,.006],[.038,.086],[.036,.089],
                   [.033,.087],[.030,.012],[.026,.009],[0,.009]];
  const shape = new THREE.LatheGeometry(profile.map(([r,h])=>new THREE.Vector2(r*scale,h*scale)),36);
  put(scene,shape,material,x,y,z);
  put(scene,new THREE.TorusGeometry(.024*scale,.005*scale,14,32),material,x+.037*scale,y+.048*scale,z);
}

function seeded(seed) {
  return () => {
    seed = (Math.imul(seed,1664525)+1013904223)>>>0;
    return seed/2**32;
  };
}

function enclosure(scene, m, y, worktop) {
  box(scene,m.floor,4,.05,4,0,y-.68,-1.1);
  box(scene,m.wall,4,2.1,.06,0,y+.20,-2.1);
  box(scene,m.wall,.06,2.1,3,-1.5,y+.20,-.86);
  box(scene,m.wall,.06,2.1,3,1.5,y+.20,-.86);
  box(scene,m.wall,3.1,.06,3,0,y+1.28,-.86);
  box(scene,worktop,1.56,.045,1.02,0,y-.0225,-.92);
  for (const x of [-.69,.69]) for (const z of [-1.37,-.48])
    box(scene,m.wood,.055,.66,.055,x,y-.35,z);
}

function window(scene,m,x,y,width) {
  box(scene,m.wood,width+.05,.47,.042,x,y,-2.052);
  box(scene,m.window,width,.415,.025,x,y,-2.025);
  box(scene,m.wood,.018,.42,.027,x,y,-2.006);
  box(scene,m.wood,width,.018,.027,x,y,-2.006);
  box(scene,m.wall,width+.09,.020,.12,x,y-.23,-1.99);
}

function desk(scene, m, y, rng) {
  enclosure(scene,m,y,m.wood);
  window(scene,m,-.70,y+.72,.57);
  box(scene,m.fabric,.66,.003,.40,-.08,y+.002,-.57);
  const book = box(scene,m.paper,.17,.017,.23,-.33,y+.013,-.83);
  book.rotation.y = -.32;
  box(scene,m.plastic,.24,.012,.34,.25,y+.014,-1.03).rotation.y=.18;
  const screen=box(scene,m.plastic,.25,.19,.014,.25,y+.110,-1.19);
  screen.rotation.x=-.13;
  box(scene,m.enamel,.21,.148,.004,.25,y+.111,-1.178).rotation.x=-.13;
  cup(scene,m.ceramic,.42,y,-.66);
  for(let i=0;i<5;i++)
    box(scene,m.ink,.07,.007,.007,-.41+i*.014,y+.007,-.64+rng()*.02).rotation.y=.56;
  box(scene,m.wood,1.1,.03,.19,0,y+.40,-1.91);
  for(let i=0;i<12;i++)
    box(scene,i%2?m.paper:m.plastic,.041,.14+rng()*.1,.11,-.44+i*.055,y+.50,-1.95);
}

function workbench(scene, m, y, rng) {
  enclosure(scene,m,y,m.concrete);
  box(scene,m.enamel,.74,.10,.22,.37,y+.84,-1.74);
  box(scene,m.window,.70,.015,.064,.37,y+.78,-1.66);
  box(scene,m.wood,.38,.028,.29,-.23,y+.015,-.67).rotation.y=.23;
  box(scene,m.plastic,.32,.08,.19,.38,y+.04,-.94);
  for(let i=0;i<12;i++){
    const x=.23+rng()*.28,z=-1.09+rng()*.18;
    const bolt=put(scene,new THREE.CylinderGeometry(.007,.007,.042,8),m.steel,x,y+.008,z);
    bolt.rotation.set(Math.PI/2,0,rng()*Math.PI);
  }
  for(let i=0;i<2;i++){
    const handle=put(scene,new THREE.CylinderGeometry(.013,.017,.14,16),m.plastic,-.37+i*.065,y+.026,-.56);
    handle.rotation.set(Math.PI/2,0,i*.5);
  }
  box(scene,m.wood,1.32,.57,.032,0,y+.31,-1.44);
  const hole = new THREE.CircleGeometry(.006,12);
  // Dark recessed pegboard holes share a mesh and material.
  const pegs = new THREE.InstancedMesh(hole,m.ink,25*11);
  const matrix = new THREE.Matrix4();
  for(let column=0;column<25;column++)for(let row=0;row<11;row++)
    pegs.setMatrixAt(column*11+row,matrix.makeTranslation(-.59+column*.049,y+.073+row*.047,-1.419));
  scene.add(pegs);
}

function kitchen(scene, m, y, rng) {
  enclosure(scene,m,y,m.concrete);
  window(scene,m,.50,y+.66,.56);
  box(scene,m.wood,.66,.03,.22,-.41,y+.58,-1.73);
  for(let i=0;i<4;i++)cup(scene,i%2?m.ceramic:m.enamel,-.61+i*.12,y+.596,-1.74,.8);
  for(let row=0;row<4;row++)for(let column=0;column<11;column++)
    box(scene,(row+column)%5===0?m.enamel:m.tile,.136,.096,.026,-.73+column*.144,y+.06+row*.103,-1.46);
  box(scene,m.wood,.34,.019,.28,-.28,y+.01,-.57).rotation.y=.23;
  const plates=[[0,0],[.074,0],[.103,.015],[.106,.019],[.103,.023],[.075,.012],[0,.009]];
  put(scene,new THREE.LatheGeometry(plates.map(point=>new THREE.Vector2(...point)),48),m.ceramic,.25,y,-.64);
  cup(scene,m.ceramic,-.46,y,-.89,1.15);
  cup(scene,m.enamel,.34,y,-.98,.87);
  for(let i=0;i<3;i++){
    const spoon = put(scene,new THREE.SphereGeometry(1,18,12),m.steel,.05+i*.033,y+.011,-.84+rng()*.02);
    spoon.scale.set(.012,.003,.023);
    box(scene,m.steel,.006,.003,.09,spoon.position.x,y+.01,spoon.position.z+.056);
  }
  box(scene,m.enamel,1.4,.32,.03,0,y-.22,-.422);
  for(let i=0;i<4;i++)
    box(scene,m.steel,.10,.014,.025,-.51+i*.34,y-.11,-.399);
}

async function foundationRoom(layout, base, assetKey) {
  const {model, anchor, yaw, scale, height_offset=0} = layout.foundation;
  const key = assetKey || model;
  if (!foundationModels.has(key)) {
    foundationModels.set(key, (async () => {
      const {GLTFLoader}=await import('/three/examples/jsm/loaders/GLTFLoader.js');
      const loader=new GLTFLoader();
      // Recognize Foundation's authored area lights. BuildScene supplies the capture lighting.
      loader.register(()=>({name:'EXT_lights_area'}));
      const gltf=await loader.loadAsync(`${base}/foundation/${model}/${model}.gltf`);
      const camerasAndLights=[];
      gltf.scene.traverse(object=>{
        if (object.isCamera || object.isLight) camerasAndLights.push(object);
        if (object.isMesh) {
          object.userData.foundationMesh=true;
          const surfaces=Array.isArray(object.material) ? object.material : [object.material];
          object.castShadow=surfaces.some(material=>material.visible && !(material.transmission>=.85));
          object.receiveShadow=true;
        }
      });
      for (const object of camerasAndLights) object.removeFromParent();
      return gltf.scene;
    })().catch(error => { foundationModels.delete(key); throw error; }));
  }
  const imported = new THREE.Group();
  const loaded=foundationModels.get(key);
  const modelScene=await loaded;
  foundationModels.delete(key);
  foundationModels.set(key,loaded);
  imported.add(modelScene.clone(true));
  imported.scale.setScalar(scale);
  imported.rotation.y=yaw;
  const offset=new THREE.Vector3().fromArray(anchor).multiplyScalar(scale).applyQuaternion(imported.quaternion);
  imported.position.set(0,layout.height+height_offset,-.41).sub(offset);
  return imported;
}

// Called after the viewport has detached the previous scene. Foundation clones
// share their geometries, materials and textures with the cached GLTF.
export async function trimViewportRooms() {
  while (foundationModels.size > 2) {
    const [key, loaded] = foundationModels.entries().next().value;
    foundationModels.delete(key);
    const resources = new Set();
    (await loaded).traverse(object => {
      if (!object.isMesh) return;
      resources.add(object.geometry);
      for (const material of [object.material].flat()) {
        resources.add(material);
        for (const value of Object.values(material)) if (value?.isTexture) resources.add(value);
      }
    });
    for (const resource of resources) resource.dispose();
  }
}

export async function buildScene(materials, layout, lighting, base = '', assetKey) {
  const scene = new THREE.Scene();
  const profile=layout.foundation?.light ?? layout.name;
  scene.background = new THREE.Color(profile==='courtyard'?0xa4bad0:0x575953);
  if (profile!=='courtyard') {
    scene.environment = materials.indoors[profile==='workbench'?'workbench':'desk'].texture;
    scene.environmentIntensity = .53*lighting.fill;
  }
  const room = new THREE.Group();
  room.quaternion.fromArray(layout.rotation);
  room.position.fromArray(layout.origin);
  scene.environmentRotation.setFromQuaternion(room.quaternion);
  scene.environmentRotation.y+=lighting.yaw;
  scene.add(room);
  if (layout.foundation) {
    room.add(await foundationRoom(layout, base, assetKey));
  } else {
    const furnish = {desk, workbench, kitchen}[layout.name];
    furnish(room,materials,layout.height,seeded(layout.seed));
  }
  if (lighting.fixture==='sun') {
    const sky = new THREE.HemisphereLight(0xe3efff,0x83735b,lighting.fill*1.5);
    const sun = new THREE.DirectionalLight(0xffe6c4,lighting.key);
    sun.position.set(6*Math.sin(lighting.yaw), 9, -4*Math.cos(lighting.yaw));
    sun.target.position.set(0,0,-.55);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048,2048);
    Object.assign(sun.shadow.camera,{left:-5,right:5,bottom:-5,top:5,near:.1,far:25});
    sun.shadow.camera.updateProjectionMatrix();
    sun.shadow.normalBias=.005;
    room.add(sky,sun,sun.target);
    return scene;
  }
  const y=layout.height;
  const settings={
    window:{place:[-.68,y+.92,-.44], tint:0xe6f1ff, strength:2.4, width:.94, height:.66},
    ceiling:{place:[.16,y+1.05,-.52], tint:0xfff4e4, strength:2.7, width:.86, height:.36},
    task:{place:[.44,y+.66,-.48], tint:0xffe7d4, strength:1.65, width:.48, height:.29},
  }[lighting.fixture];
  const key = new THREE.SpotLight(settings.tint,lighting.key*settings.strength,3.6,.89,.61,2);
  key.target.position.set(0,y+.14,-.55);
  key.position.set(...settings.place);
  key.position.x+=Math.sin(lighting.yaw)*.16;
  key.position.y+=(lighting.elevation-.66)*.36;
  key.castShadow = true;
  key.shadow.mapSize.set(1024,1024);
  Object.assign(key.shadow.camera,{near:.15,far:3.6});
  key.shadow.camera.updateProjectionMatrix();
  key.shadow.radius=2.4;
  key.shadow.normalBias=.00115;
  key.shadow.bias=-.00012;
  const fill=new THREE.RectAreaLight(settings.tint,lighting.fill*1.65,settings.width,settings.height);
  fill.position.copy(key.position);
  fill.position.x-=.16;
  fill.position.y-=.14;
  fill.position.z+=.25;
  fill.lookAt(key.target.position);
  room.add(key,key.target,fill);
  return scene;
}
