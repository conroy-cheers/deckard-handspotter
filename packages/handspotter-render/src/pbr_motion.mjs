import * as THREE from 'three';

const radians = Math.PI / 180;
const vec = values => new THREE.Vector3(...values);

function segmentDistanceSquared(point, origin, segment, t, scratch) {
  return point.distanceToSquared(scratch.copy(origin).addScaledVector(segment, THREE.MathUtils.clamp(t, 0, 1)));
}

export function bindArmVelocities(mesh, arms) {
  if (mesh.geometry.hasAttribute('armVelocity')) return;
  const paths = Object.values(arms ?? {}).map(arm => {
    const [shoulder, elbow, wrist, palm] = [arm.shoulder, arm.elbow, arm.wrist, arm.palm].map(vec);
    const upper = elbow.clone().sub(shoulder);
    const lower = wrist.clone().sub(elbow);
    const hand = palm.clone().sub(wrist);
    return {shoulder, elbow, wrist, upper, lower, hand, omega:vec(arm.rotation_deg).multiplyScalar(radians)};
  });
  if (!paths.some(path => path.omega.lengthSq())) return;

  const xyz = mesh.geometry.attributes.position;
  const velocities = new Float32Array(3 * xyz.count);
  const point = new THREE.Vector3();
  const relative = new THREE.Vector3();
  const sample = new THREE.Vector3();
  const velocity = new THREE.Vector3();
  for (let i = 0; i < xyz.count; i++) {
    point.fromBufferAttribute(xyz, i);
    velocity.set(0, 0, 0);
    let influence = 0, total = 0;
    for (const {shoulder, elbow, wrist, upper, lower, hand, omega} of paths) {
      const upperT = relative.subVectors(point, shoulder).dot(upper) / upper.lengthSq();
      const lowerT = relative.subVectors(point, elbow).dot(lower) / lower.lengthSq();
      const handT = relative.subVectors(point, wrist).dot(hand) / hand.lengthSq();
      const upperDist = segmentDistanceSquared(point, shoulder, upper, upperT, sample);
      const distalDist = Math.min(segmentDistanceSquared(point, elbow, lower, lowerT, sample),
                                  segmentDistanceSquared(point, wrist, hand, handT, sample));
      const distance = Math.sqrt(Math.min(upperDist, distalDist));
      const shoulderFade = upperDist < distalDist ? THREE.MathUtils.smoothstep(upperT, 0, .32) : 1;
      const support = shoulderFade * (1 - THREE.MathUtils.smoothstep(distance, .054, .117));
      influence = Math.max(influence, support);
      // Blend at nearby limbs and at the shoulder seam. Duplicated vertices
      // in baked skin and ID groups get the same velocity from their position.
      const weight = support * Math.exp(-800 * distance * distance);
      sample.crossVectors(omega, relative.subVectors(point, shoulder));
      velocity.addScaledVector(sample, weight);
      total += weight;
    }
    if (total) velocity.multiplyScalar(influence / total).toArray(velocities, i * 3);
  }
  mesh.geometry.setAttribute('armVelocity', new THREE.BufferAttribute(velocities, 3));
}

export function cameraMotion(settings, pivot) {
  return {
    angular: vec(settings?.rotation_deg ?? [0, 0, 0]).multiplyScalar(radians),
    translation: vec(settings?.translation_m ?? [0, 0, 0]),
    pivot: vec(pivot),
  };
}

export function motionMaterial(mask, moving, camera) {
  const material = new THREE.MeshBasicMaterial({
    color:0, map:mask.map ?? null, alphaMap:mask.alphaMap ?? null,
    alphaTest:mask.alphaTest, side:mask.side, toneMapped:false,
    blending:THREE.NoBlending,
  });
  material.visible = mask.visible;
  material.customProgramCacheKey = () => moving ? 'capture-arm-velocity' : 'capture-rig-velocity';
  material.onBeforeCompile = shader => {
    shader.uniforms.captureAngular = {value:camera.angular};
    shader.uniforms.captureTranslation = {value:camera.translation};
    shader.uniforms.capturePivot = {value:camera.pivot};
    shader.uniforms.capturePart = {value:1 + (mask.userData.capturePart ?? 0)};
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        ${moving ? 'attribute vec3 armVelocity;' : ''}
        varying vec3 capturePosition;
        varying vec3 captureObjectVelocity;`)
      .replace('#include <project_vertex>', `#include <project_vertex>
        vec4 captureWorld = vec4(transformed, 1.0);
        #ifdef USE_BATCHING
          captureWorld = batchingMatrix * captureWorld;
        #endif
        #ifdef USE_INSTANCING
          captureWorld = instanceMatrix * captureWorld;
        #endif
        capturePosition = (modelMatrix * captureWorld).xyz;
        captureObjectVelocity = ${moving ? 'mat3(modelMatrix) * armVelocity' : 'vec3(0.0)'};`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform vec3 captureAngular, captureTranslation, capturePivot;
        uniform float capturePart;
        varying vec3 capturePosition;
        varying vec3 captureObjectVelocity;`)
      .replace('#include <dithering_fragment>', `
        vec3 relativeVelocity = captureObjectVelocity - captureTranslation -
            cross(captureAngular, capturePosition - capturePivot);
        float movingArm = step(0.001, length(captureObjectVelocity));
        gl_FragColor = vec4(relativeVelocity / length(capturePosition - cameraPosition),
                            capturePart + 4.0 * movingArm);
      `);
  };
  return material;
}
