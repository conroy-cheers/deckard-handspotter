import * as THREE from 'three/webgpu';
import { Fn, If, instanceIndex, instancedArray, uint, uniform, vec3, vec4 } from 'three/tsl';
import { SkinCollider } from './skin_collider.mjs';

// A batch contains no two edges that write to the same vertex. Later
// dispatches see the corrections from earlier batches in the same sweep.
function colorEdges(count, edges, lengths) {
  const incident = Array.from({length:count}, () => new Set());
  const batches = [];
  for (let edge=0; edge<edges.length; edge++) {
    const [a,b] = edges[edge];
    let color = 0;
    while (incident[a].has(color) || incident[b].has(color)) color++;
    (batches[color] ??= []).push(a,b,lengths[edge],0);
    incident[a].add(color);
    incident[b].add(color);
  }
  return batches;
}

async function settleCloth(renderer, garment, contact) {
  const count = garment.position.length/3;
  const start = new Float32Array(count*4);
  for (let i=0; i<count; i++) {
    const src = 3*i, dst = 4*i;
    start[dst] = garment.position[src];
    start[dst+1] = garment.position[src+1];
    start[dst+2] = garment.position[src+2];
    start[dst+3] = 1-garment.pin[i];
  }
  const anchor = instancedArray(start.slice(),'vec4').toReadOnly();
  const positions = instancedArray(start,'vec4');
  const previous = instancedArray(start.slice(),'vec4');
  const contactPlanes = instancedArray(count,'vec4');
  const constraints = [];
  const batches = colorEdges(count,garment.edges,garment.rest_lengths);
  const edges = instancedArray(new Float32Array(batches.flat()),'vec4').toReadOnly();
  function displace(vertex, index, delta) {
    const plane = contactPlanes.element(index);
    const surface = vertex.xyz.add(delta).toVar();
    // Project onto the most recent skin half-space while solving an edge.
    // Otherwise each distance pass can move a cloth vertex through skin.
    vertex.xyz.assign(surface.add(plane.xyz.mul(plane.w.sub(surface.dot(plane.xyz)).max(0))));
  }

  const strain = .03 * .23 / garment.stiffness;
  let first = 0;
  for (const batch of batches) {
    const size = batch.length/4;
    const offset = uniform(first,'uint');
    constraints.push(Fn(() => {
      If(instanceIndex.lessThan(uint(size)), () => {
        const edge = edges.element(instanceIndex.add(offset)).toVar();
        const a = positions.element(uint(edge.x));
        const b = positions.element(uint(edge.y));
        const direction = b.xyz.sub(a.xyz).toVar();
        const length = direction.length().max(1e-6).toVar();
        const rest = edge.z;
        // Small in-plane compliance with independent freedom to fold
        // across a seam. Springs alone cannot give a strain ceiling.
        const error = length.sub(length.clamp(rest.mul(1-strain),rest.mul(1+strain)));
        const correction = direction.mul(error.clamp(-.016,.016).div(length))
          .div(a.w.add(b.w).max(1e-6)).toVar();
        displace(a,uint(edge.x),correction.mul(a.w));
        displace(b,uint(edge.y),correction.mul(b.w).negate());
      });
    })().compute(size).setName('Cloth distance constraints'));
    first += size;
  }

  const predict = Fn(() => {
    If(instanceIndex.lessThan(uint(count)), () => {
      const point = positions.element(instanceIndex);
      const last = previous.element(instanceIndex);
      const seed = anchor.element(instanceIndex);
      const velocity = point.xyz.sub(last.xyz).mul(.4).toVar();
      const speed = velocity.length().max(1e-6).toVar();
      const tether = seed.xyz.sub(point.xyz).mul(seed.w.oneMinus().mul(.04));
      last.xyz.assign(point.xyz);
      point.xyz.addAssign(velocity.mul(speed.min(.016).div(speed))
        .add(vec3(0,-9.81/(90*90),0)).add(tether).mul(point.w));
    });
  })().compute(count).setName('Predict cloth positions');

  const contactMargin = .006;
  const collide = Fn(() => {
    If(instanceIndex.lessThan(uint(count)), () => {
      const point = positions.element(instanceIndex);
      If(point.w.greaterThan(0), () => {
        const plane = contactPlanes.element(instanceIndex);
        const push = contact(point.xyz,contactMargin).toVar();
        const length = push.length().toVar();
        If(length.greaterThan(1e-6), () => {
          const normal = push.div(length).toVar();
          const gap = length.sub(contactMargin).min(.008).toVar();
          point.xyz.addAssign(normal.mul(gap.max(0)));
          plane.assign(vec4(normal,point.xyz.dot(normal).add(gap.min(0))));
        }).Else(() => plane.assign(vec4(0)));
      });
    });
  })().compute(count).setName('Resolve cloth skin contact');

  const resetVelocity = Fn(() => {
    If(instanceIndex.lessThan(uint(count)), () => {
      previous.element(instanceIndex).assign(positions.element(instanceIndex));
    });
  })().compute(count).setName('Reset cloth velocity after fitting');

  const sweep = [...constraints,...constraints,...constraints,collide];
  try {
    renderer.compute(collide);
    for (let sweepIndex=0; sweepIndex<60; sweepIndex++) renderer.compute(sweep);
    renderer.compute(resetVelocity);
    const step = [predict,...sweep,...sweep,...sweep];
    for (let frame=0; frame<32; frame++) renderer.compute(step);
    for (let sweepIndex=0; sweepIndex<120; sweepIndex++) renderer.compute(sweep);
    const packed = new Float32Array(await renderer.getArrayBufferAsync(positions.value));
    const out = new Float32Array(count*3);
    for (let i=0; i<count; i++) out.set(packed.subarray(4*i,4*i+3),3*i);
    if (!out.every(Number.isFinite)) throw new Error('Cloth distance solver returned non-finite positions');
    return out;
  } finally {
    for (const node of [...constraints,predict,collide,resetVelocity]) node.dispose();
    for (const buffer of [edges,anchor,positions,previous,contactPlanes]) {
      buffer.value.dispatchEvent({type:'dispose'});
    }
  }
}

// GPU contacts operate at particles. A coarse sleeve triangle can still cut
// through a bent arm between three clear vertices. Project those residual
// contacts on the posed skin before sharing the mesh with the camera views.
function finishContacts(position, faces, distance) {
  const corrections=new Float32Array(position.length);
  const weights=new Uint32Array(position.length/3);
  const moved=new Uint8Array(weights.length).fill(1);
  const point=new THREE.Vector3();
  function project(a,b,c) {
    if (!(moved[a] || moved[b] || moved[c])) return;
    point.set((position[3*a]+position[3*b]+position[3*c])/3,
              (position[3*a+1]+position[3*b+1]+position[3*c+1])/3,
              (position[3*a+2]+position[3*b+2]+position[3*c+2])/3);
    const gap=distance.sample(point);
    if (gap>=-.001) return;
    point.subVectors(distance.hit.point,point).multiplyScalar(Math.min(.008,.002-gap)/-gap);
    for (const index of [a,b,c]) {
      corrections[3*index]+=point.x;
      corrections[3*index+1]+=point.y;
      corrections[3*index+2]+=point.z;
      weights[index]++;
    }
  }
  for (let pass=0;pass<24;pass++) {
    corrections.fill(0);
    weights.fill(0);
    for (let i=0;i<weights.length;i++) project(i,i,i);
    for (let i=0;i<faces.length;i+=3) project(faces[i],faces[i+1],faces[i+2]);
    if (!weights.some(Boolean)) break;
    moved.fill(0);
    for (let i=0;i<weights.length;i++) {
      if (!weights[i]) continue;
      moved[i]=1;
      for (let axis=0;axis<3;axis++) position[3*i+axis]+=corrections[3*i+axis]/weights[i];
    }
  }
}

function clearance(surface,faces,distance){
  let minimum=0, deeper=0, count=0;
  const point=new THREE.Vector3();
  function sample(){
    const mm=1000*distance.sample(point);
    minimum=Math.min(minimum,mm);
    deeper+=mm< -5;
    count++;
  }
  for(let i=0;i<surface.length;i+=3){
    point.fromArray(surface,i);
    sample();
  }
  for(let i=0;i<faces.length;i+=3){
    const a=3*faces[i],b=3*faces[i+1],c=3*faces[i+2];
    point.set((surface[a]+surface[b]+surface[c])/3,
      (surface[a+1]+surface[b+1]+surface[c+1])/3,
      (surface[a+2]+surface[b+2]+surface[c+2])/3);
    sample();
  }
  return {samples:count,inward_5mm:deeper,max_inward_mm:-minimum};
}

export async function wardrobeRenderer(){
  const renderer=new THREE.WebGPURenderer();
  const distance = new SkinCollider();
  let gpuError;
  await renderer.init();
  if(!renderer.backend.isWebGPUBackend) throw new Error('SOMA-X cloth draping requires WebGPU');
  renderer.backend.device.addEventListener('uncapturederror',e=>{gpuError=e.error.message;});
  await distance.preload(renderer);
  return {
    async dress(garment,surfaces){
      const start=performance.now();
      distance.update(surfaces);
      const contact=distance.gpuContact(renderer,garment);
      const solveStart=performance.now();
      const position=await settleCloth(renderer,garment,contact);
      finishContacts(position,garment.faces,distance);
      let squaredMotion=0;
      for (let i=0;i<position.length;i++) squaredMotion+=(position[i]-garment.position[i])**2;
      const solved=performance.now();
      if(gpuError) throw new Error(gpuError);
      return {
        surface:{material:'cloth',part:3,position,faces:garment.faces},
        statistics:{...clearance(position,garment.faces,distance),
          rms_displacement_mm:1000*Math.sqrt(squaredMotion/(position.length/3)),
          vertices:position.length/3,collider_ms:solveStart-start,solve_ms:solved-solveStart},
      };
    },
    dispose(){ distance.dispose(); renderer.dispose(); },
  };
}
