typedef unsigned int uint;

typedef struct { float x, y, z, meta; } Slot;
typedef struct { double x, y, z; } Vec3;

static float minf(float a, float b) { return a < b ? a : b; }
static float maxf(float a, float b) { return a > b ? a : b; }

void refit(Slot *slots, uint node_count, uint vertex_offset, const float *positions, uint vertex_count) {
    Slot *vertices = slots + vertex_offset;
    const Slot *topology = vertices + vertex_count;
    for (uint v = 0; v < vertex_count; ++v) {
        vertices[v] = (Slot){positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2], 0};
    }

    for (uint node = node_count; node-- > 0;) {
        Slot *lo = slots + node * 2;
        Slot *hi = lo + 1;
        const uint count = (uint)hi->meta;
        if (count) {
            const uint first_triangle = (uint)lo->meta;
            const Slot *triangles = topology + first_triangle;
            lo->x = lo->y = lo->z = __builtin_inff();
            hi->x = hi->y = hi->z = -__builtin_inff();
            for (uint t = 0; t < count; ++t) {
                const Slot triangle = triangles[t];
                const uint indices[] = {(uint)triangle.x, (uint)triangle.y, (uint)triangle.z};
                for (uint v = 0; v < 3; ++v) {
                    const Slot p = slots[indices[v]];
                    lo->x = minf(lo->x, p.x); lo->y = minf(lo->y, p.y); lo->z = minf(lo->z, p.z);
                    hi->x = maxf(hi->x, p.x); hi->y = maxf(hi->y, p.y); hi->z = maxf(hi->z, p.z);
                }
            }
        } else {
            const Slot *left = hi + 1;
            const uint right_node = left[1].meta != 0 ? node + 2 : (uint)left[0].meta;
            const Slot *right = slots + 2 * right_node;
            lo->x = minf(left[0].x, right[0].x);
            lo->y = minf(left[0].y, right[0].y);
            lo->z = minf(left[0].z, right[0].z);
            hi->x = maxf(left[1].x, right[1].x);
            hi->y = maxf(left[1].y, right[1].y);
            hi->z = maxf(left[1].z, right[1].z);
        }
    }
}

static Vec3 xyz(Slot p) { return (Vec3){p.x, p.y, p.z}; }
static Vec3 add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3 sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec3 mul(Vec3 a, double k) { return (Vec3){a.x * k, a.y * k, a.z * k}; }
static double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 cross(Vec3 a, Vec3 b) {
    return (Vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static Vec3 segment_point(Vec3 p, Vec3 a, Vec3 b) {
    const Vec3 dir = sub(b, a);
    const double length2 = dot(dir, dir);
    if (length2 == 0) return a;
    const double t = dot(sub(p, a), dir) / length2;
    return t <= 0 ? a : t >= 1 ? b : add(a, mul(dir, t));
}

static Vec3 triangle_point(Vec3 p, Vec3 a, Vec3 b, Vec3 c, Vec3 n, double normal2) {
    const Vec3 q = sub(p, mul(n, dot(sub(p, a), n) / normal2));
    if (dot(cross(sub(b, a), sub(q, a)), n) >= 0 &&
        dot(cross(sub(c, b), sub(q, b)), n) >= 0 &&
        dot(cross(sub(a, c), sub(q, c)), n) >= 0) {
        return q;
    }

    const Vec3 ab = segment_point(p, a, b), bc = segment_point(p, b, c), ca = segment_point(p, c, a);
    const Vec3 ab_delta = sub(p, ab), bc_delta = sub(p, bc), ca_delta = sub(p, ca);
    const double ab2 = dot(ab_delta, ab_delta), bc2 = dot(bc_delta, bc_delta), ca2 = dot(ca_delta, ca_delta);
    return ab2 <= bc2 && ab2 <= ca2 ? ab : bc2 <= ca2 ? bc : ca;
}

static double axis_distance(double p, double lower, double upper) {
    return p < lower ? lower - p : p > upper ? p - upper : 0.0;
}

static double bounds_distance2(const Slot *lower, Vec3 p) {
    const Slot lo = lower[0], hi = lower[1];
    const Vec3 delta = {axis_distance(p.x, lo.x, hi.x), axis_distance(p.y, lo.y, hi.y),
                        axis_distance(p.z, lo.z, hi.z)};
    return dot(delta, delta);
}

void correct_anchors(const Slot *slots, uint triangle_offset, float *anchors, uint count, double thickness) {
    typedef struct { uint node; double distance2; } Candidate;
    // MeshBVH uses a maximum depth of 40. Only the farther sibling at each
    // depth can be pending while the nearer subtree is visited.
    Candidate pending[41];
    for (uint i = 0; i < count; ++i) {
        float *anchor = anchors + i * 3;
        const Vec3 p = {anchor[0], anchor[1], anchor[2]};
        Vec3 normal = {0, 0, 0};
        int inside = 0;
        double best = __builtin_inf();
        uint remaining = 1;
        pending[0] = (Candidate){0, 0};
        while (remaining) {
            const Candidate candidate = pending[--remaining];
            if (candidate.distance2 >= best) continue;
            const uint node = candidate.node;
            const Slot lo = slots[node * 2];
            const Slot hi = slots[node * 2 + 1];
            const uint leaf_count = (uint)hi.meta;
            if (!leaf_count) {
                const uint left = node + 1;
                const uint right = slots[2 * left + 1].meta != 0 ? left + 1 : (uint)slots[2 * left].meta;
                const Candidate a = {left, bounds_distance2(slots + 2 * left, p)};
                const Candidate b = {right, bounds_distance2(slots + 2 * right, p)};
                const Candidate near = a.distance2 < b.distance2 ? a : b;
                const Candidate far = a.distance2 < b.distance2 ? b : a;
                if (far.distance2 < best) pending[remaining++] = far;
                if (near.distance2 < best) pending[remaining++] = near;
                continue;
            }

            const Slot *triangles = slots + triangle_offset + (uint)lo.meta;
            for (uint t = 0; t < leaf_count; ++t) {
                const Slot face = triangles[t];
                const Vec3 a = xyz(slots[(uint)face.x]), b = xyz(slots[(uint)face.y]), c = xyz(slots[(uint)face.z]);
                const Vec3 n = cross(sub(b, a), sub(c, a));
                const double normal2 = dot(n, n);
                if (normal2 < 1e-24) continue;
                const Vec3 delta = sub(p, triangle_point(p, a, b, c, n, normal2));
                const double distance2 = dot(delta, delta);
                if (distance2 < best) {
                    best = distance2;
                    normal = mul(n, 1 / __builtin_sqrt(normal2));
                    inside = dot(delta, normal) < 0;
                }
            }
        }
        const double distance = __builtin_sqrt(best) * (inside ? -1 : 1);
        if (distance < thickness) {
            const Vec3 corrected = add(p, mul(normal, thickness - distance));
            anchor[0] = corrected.x;
            anchor[1] = corrected.y;
            anchor[2] = corrected.z;
        }
    }
}
