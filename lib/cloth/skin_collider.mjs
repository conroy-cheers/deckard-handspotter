import * as THREE from 'three/webgpu';
import { Break, Fn, If, Loop, float, instancedArray, uint, vec3, wgslFn } from 'three/tsl';
import { MeshBVH } from '/cloth-deps/mesh-bvh.mjs';

// Contact queries visit only nodes within this distance of the cloth vertex.
// The band covers the solver's 10 mm step cap plus the contact thickness.
const band = .045;
const thickness = .006;

const segmentPoint = wgslFn(`
fn skinSegmentPoint(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
  let d = b - a;
  return a + d * clamp(dot(p - a, d) / max(dot(d, d), 1e-16), 0.0, 1.0);
}
`);

const trianglePoint = wgslFn(`
fn skinTrianglePoint(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  let n = cross(b - a, c - a);
  let q = p - n * (dot(p - a, n) / max(dot(n, n), 1e-24));
  if (dot(cross(b - a, q - a), n) >= 0.0 &&
      dot(cross(c - b, q - b), n) >= 0.0 &&
      dot(cross(a - c, q - c), n) >= 0.0 && dot(n, n) > 1e-24) {
    return q;
  }
  let ab = skinSegmentPoint(p, a, b);
  let bc = skinSegmentPoint(p, b, c);
  let ca = skinSegmentPoint(p, c, a);
  let ab2 = dot(p - ab, p - ab);
  let bc2 = dot(p - bc, p - bc);
  let ca2 = dot(p - ca, p - ca);
  if (ab2 <= min(bc2, ca2)) { return ab; }
  return select(ca, bc, bc2 <= ca2);
}
`, [segmentPoint]);

function colliderLayout(tree, indices, vertexCount) {
  // Nodes occupy two vec4s: xyz holds bounds, w holds a triangle range for
  // leaves or a subtree end and zero for internal nodes. Posed vertices and
  // immutable triangle indices follow the nodes.
  const layout = [], bounds = [], ancestors = [];
  tree.traverse((depth, leaf, nodeBounds, offset, count) => {
    while (ancestors.length > depth) layout[ancestors.pop() * 2] = layout.length / 2;
    if (!leaf) ancestors.push(layout.length / 2);
    layout.push(leaf ? offset : 0, leaf ? count : 0);
    bounds.push(nodeBounds);
  });
  const nodeCount = layout.length / 2;
  for (const node of ancestors) layout[node * 2] = nodeCount;
  const vertexOffset = nodeCount * 2;
  const triangles = vertexOffset + vertexCount;
  const data = new Float32Array((triangles + indices.length / 3) * 4);
  for (let node = 0; node < nodeCount; node++) {
    data[node * 8 + 3] = layout[node * 2];
    data[node * 8 + 7] = layout[node * 2 + 1];
  }
  for (let i = 0, target = triangles * 4; i < indices.length; i += 3, target += 4) {
    data[target] = indices[i] + vertexOffset;
    data[target + 1] = indices[i + 1] + vertexOffset;
    data[target + 2] = indices[i + 2] + vertexOffset;
  }
  return {data, bounds, nodeCount, vertexOffset, triangles};
}

async function optionalAsset(path) {
  const response = await fetch(path);
  if (response.status === 404) return;
  if (!response.ok) throw new Error(`Could not load ${path}: ${response.status}`);
  return response.arrayBuffer();
}

class ColliderKernel {
  constructor(exports) {
    this.exports = exports;
    this.heap = Number(exports.__heap_base.value);
  }

  reserve(bytes) {
    const pages = Math.ceil((this.heap + bytes - this.exports.memory.buffer.byteLength) / 65536);
    if (pages > 0) this.exports.memory.grow(pages);
  }

  data(layout) {
    return new Float32Array(this.exports.memory.buffer, this.heap, layout.data.length);
  }

  prepareLayout(layout, vertices) {
    // Reserve a typical garment's anchor scratch along with the fixed
    // collider arrays. Larger garments can grow it without changing bindings.
    this.reserve(layout.data.byteLength + vertices.byteLength + 65536);
    this.data(layout).set(layout.data);
  }

  preparePose(layout, vertices, garment) {
    const positions = this.heap + layout.data.byteLength;
    const anchorOffset = positions + vertices.byteLength;
    const indicesOffset = anchorOffset + garment.pin.length * 12;
    this.reserve(indicesOffset - this.heap + garment.pin.length * 4);
    const buffer = this.exports.memory.buffer;
    new Float32Array(buffer, positions, vertices.length).set(vertices);
    this.exports.refit(this.heap, layout.nodeCount, layout.vertexOffset, positions, vertices.length / 3);

    // Batch exact nearest-triangle queries for the fixed neckline vertices.
    const anchors = new Float32Array(buffer, anchorOffset, garment.pin.length * 3);
    const indices = new Uint32Array(buffer, indicesOffset, garment.pin.length);
    let count = 0;
    for (let i = 0; i < garment.pin.length; i++) {
      if (garment.pin[i] !== 1) continue;
      const src = i * 3, dst = count * 3;
      indices[count++] = src;
      anchors[dst] = garment.position[src];
      anchors[dst + 1] = garment.position[src + 1];
      anchors[dst + 2] = garment.position[src + 2];
    }
    this.exports.correct_anchors(this.heap, layout.triangles, anchorOffset, count, thickness);
    for (let i = 0; i < count; i++) {
      const dst = indices[i], src = i * 3;
      garment.position[dst] = anchors[src];
      garment.position[dst + 1] = anchors[src + 1];
      garment.position[dst + 2] = anchors[src + 2];
    }
  }
}

export class SkinCollider {
  constructor() {
    this.hit = {};
    this.normal = new THREE.Vector3();
    this.a = new THREE.Vector3();
    this.b = new THREE.Vector3();
    this.c = new THREE.Vector3();
    this.delta = new THREE.Vector3();
    this.anchor = new THREE.Vector3();
  }

  async preload(renderer) {
    const [binary, wasm] = await Promise.all([
      optionalAsset('/skin-collider-topology.bin'), optionalAsset('/skin-collider.wasm'),
    ]);
    if (wasm) this.kernel = new ColliderKernel((await WebAssembly.instantiate(wasm)).instance.exports);
    if (!binary) return;
    const [version, components, indexCount, rootBytes] = new Uint32Array(binary, 0, 4);
    const indices = new Uint32Array(binary, 16, indexCount);
    const ordered = new Uint32Array(binary, 16 + indexCount * 4, indexCount);
    const root = binary.slice(16 + indexCount * 8);
    if (root.byteLength !== rootBytes || version !== 1) {
      throw new Error('Invalid packaged skin collider topology');
    }
    this.topology = indices;
    this.vertices = new Float32Array(components);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.vertices, 3));
    this.tree = MeshBVH.deserialize({version, index: ordered, roots: [root]}, this.geometry);
    this.indices = this.geometry.index.array;
    this.prepareLayout(renderer);
  }

  update(surfaces) {
    const skin = surfaces.filter(surface => surface.material === 'skin');
    const positionCount = skin.reduce((count, surface) => count + surface.position.length, 0);
    const indexCount = skin.reduce((count, surface) => count + surface.faces.length, 0);
    if (!indexCount) throw new Error('Cloth needs a skin collider');
    const rebuild = this.vertices?.length !== positionCount || !this.matches(skin, indexCount);
    if (rebuild) {
      this.dispose();
      this.geometry = new THREE.BufferGeometry();
      this.vertices = new Float32Array(positionCount);
      this.topology = new Uint32Array(indexCount);
      let vertexOffset = 0, indexOffset = 0;
      for (const {position, faces} of skin) {
        for (let i = 0; i < faces.length; i++) this.topology[indexOffset + i] = faces[i] + vertexOffset;
        indexOffset += faces.length;
        vertexOffset += position.length / 3;
      }
      this.geometry.setAttribute('position', new THREE.BufferAttribute(this.vertices, 3));
      this.geometry.setIndex(new THREE.BufferAttribute(this.topology.slice(), 1));
    }
    let offset = 0;
    for (const {position} of skin) {
      this.vertices.set(position, offset);
      offset += position.length;
    }
    if (rebuild) {
      this.tree = new MeshBVH(this.geometry, {targetLeafSize: 8});
      this.indices = this.geometry.index.array;
      this.layout = null;
    } else if (!this.kernel) {
      this.tree.refit();
    }
  }

  matches(skin, indexCount) {
    if (this.topology?.length !== indexCount) return false;
    let vertexOffset = 0, indexOffset = 0;
    for (const {position, faces} of skin) {
      for (let i = 0; i < faces.length; i++) {
        if (this.topology[indexOffset + i] !== faces[i] + vertexOffset) return false;
      }
      indexOffset += faces.length;
      vertexOffset += position.length / 3;
    }
    return true;
  }

  syncDiagnosticBounds() {
    // Independent MeshBVH queries are only used to score the settled cloth.
    const data = this.kernel.data(this.layout);
    for (let n = 0, o = 0; n < this.layout.nodeCount; n++, o += 8) {
      const bound = this.layout.bounds[n];
      bound[0] = data[o];
      bound[1] = data[o + 1];
      bound[2] = data[o + 2];
      bound[3] = data[o + 4];
      bound[4] = data[o + 5];
      bound[5] = data[o + 6];
    }
    this.diagnosticsDirty = false;
  }

  sample(point) {
    if (this.diagnosticsDirty) this.syncDiagnosticBounds();
    const hit = this.tree.closestPointToPoint(point, this.hit);
    const base = hit.faceIndex * 3;
    const {indices, vertices, a, b, c, normal, delta} = this;
    a.fromArray(vertices, indices[base] * 3);
    b.fromArray(vertices, indices[base + 1] * 3);
    c.fromArray(vertices, indices[base + 2] * 3);
    THREE.Triangle.getNormal(a, b, c, normal);
    return delta.subVectors(point, hit.point).dot(normal) < 0 ? -hit.distance : hit.distance;
  }

  correctAnchors(garment) {
    const anchor = this.anchor;
    for (let i = 0; i < garment.pin.length; i++) {
      if (garment.pin[i] !== 1) continue;
      anchor.fromArray(garment.position, 3 * i);
      const gap = this.sample(anchor);
      if (gap < thickness) anchor.addScaledVector(this.normal, thickness - gap).toArray(garment.position, 3 * i);
    }
  }

  prepareLayout(renderer) {
    this.layout = colliderLayout(this.tree, this.indices, this.vertices.length / 3);
    // Keep one binding across poses for OpenGL ES compatibility. Indices
    // and traversal links remain fixed while bounds and vertices change.
    this.field = instancedArray(this.layout.data, 'vec4').toReadOnly();
    this.project = this.contactFunction();
    this.kernel?.prepareLayout(this.layout, this.vertices);
    this.backend = renderer.backend;
    // Three.js copies the complete layout into the mapped storage buffer here.
    // Subsequent writes cover only the contiguous bounds and posed vertices.
    this.backend.createStorageAttribute(this.field.value);
  }

  packPose() {
    const {data, bounds, vertexOffset} = this.layout;
    for (let node = 0, i = 0; node < bounds.length; node++, i += 8) {
      const box = bounds[node];
      data[i] = box[0];
      data[i + 1] = box[1];
      data[i + 2] = box[2];
      data[i + 4] = box[3];
      data[i + 5] = box[4];
      data[i + 6] = box[5];
    }
    const vertices = this.vertices;
    for (let src = 0, dst = vertexOffset * 4; src < vertices.length; src += 3, dst += 4) {
      data[dst] = vertices[src];
      data[dst + 1] = vertices[src + 1];
      data[dst + 2] = vertices[src + 2];
    }
  }

  gpuContact(renderer, garment) {
    if (!this.layout) this.prepareLayout(renderer);
    let data;
    if (this.kernel) {
      this.kernel.preparePose(this.layout, this.vertices, garment);
      this.diagnosticsDirty = true;
      data = this.kernel.data(this.layout);
    } else {
      this.correctAnchors(garment);
      this.packPose();
      data = this.layout.data;
    }
    const {buffer} = renderer.backend.get(this.field.value);
    renderer.backend.device.queue.writeBuffer(buffer, 0, data.subarray(0, this.layout.triangles * 4));
    return this.project;
  }

  contactFunction() {
    const {nodeCount: count, triangles} = this.layout;
    const field = this.field;
    return Fn(([p,margin]) => {
      const best = float(band * band).toVar();
      const direction = vec3(0).toVar();
      const normal = vec3(0).toVar();
      const node = uint(0).toVar();
      Loop(count, () => {
        If(node.greaterThanEqual(uint(count)), () => Break());
        const lower = field.element(node.mul(2)).toVar();
        const upper = field.element(node.mul(2).add(1)).toVar();
        const outside = lower.xyz.sub(p).max(p.sub(upper.xyz)).max(0).toVar();
        const leaf = uint(upper.w).toVar();
        const next = node.add(1).toVar();
        If(outside.dot(outside).greaterThanEqual(best), () => {
          node.assign(leaf.equal(0).select(uint(lower.w), next));
        }).Else(() => {
          If(leaf.notEqual(0), () => {
            const offset = uint(lower.w).add(triangles).toVar();
            Loop(leaf, ({i}) => {
              const indices = field.element(offset.add(uint(i))).toVar();
              const a = field.element(uint(indices.x)).xyz.toVar();
              const b = field.element(uint(indices.y)).xyz.toVar();
              const c = field.element(uint(indices.z)).xyz.toVar();
              const n = b.sub(a).cross(c.sub(a)).toVar();
              const norm = n.dot(n).toVar();
              const d = p.sub(trianglePoint(p, a, b, c)).toVar();
              const squared = d.dot(d).toVar();
              If(squared.lessThan(best).and(norm.greaterThan(1e-24)), () => {
                best.assign(squared);
                direction.assign(d);
                normal.assign(n.mul(norm.inverseSqrt()));
              });
            });
          });
          node.assign(next);
        });
      });
      const distance = best.sqrt().toVar();
      const sign = direction.dot(normal).lessThan(0).select(float(-1), float(1));
      const exterior = distance.greaterThan(1e-6)
        .select(direction.mul(sign).div(distance.max(1e-6)), normal);
      return exterior.mul(float(thickness).add(margin ?? 0).sub(distance.mul(sign)).max(0));
    });
  }

  dispose() {
    this.diagnosticsDirty = false;
    if (this.field) {
      const attribute = this.field.value;
      attribute.dispatchEvent({type: 'dispose'});
      // The buffer may have been preallocated before a first compute dispatch.
      if (this.backend.get(attribute).buffer) this.backend.destroyAttribute(attribute);
    }
    this.field = null;
    this.geometry?.dispose();
  }
}
