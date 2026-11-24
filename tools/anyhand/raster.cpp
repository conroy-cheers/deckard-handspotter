// CPU rasterizer for the AnyHand pilot's pinhole cubemap faces.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {
struct Vertex {
    double x, y, z, gray;
};

Vertex mix(Vertex a, Vertex b, double t) {
    return {a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t,
            a.z + (b.z-a.z)*t, a.gray + (b.gray-a.gray)*t};
}

void triangle(Vertex a, Vertex b, Vertex c, uint8_t part, int size,
              float *depth, float *gray, uint8_t *labels) {
    const double center = (size-1)*.5;
    for (Vertex *p : {&a, &b, &c}) {
        p->x = p->x/p->z*center + center;
        p->y = p->y/p->z*center + center;
    }
    const double area = (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    if (std::abs(area) < 1e-8) return;
    const int xmin = std::max(0, int(std::ceil(std::min({a.x,b.x,c.x}))));
    const int xmax = std::min(size-1, int(std::floor(std::max({a.x,b.x,c.x}))));
    const int ymin = std::max(0, int(std::ceil(std::min({a.y,b.y,c.y}))));
    const int ymax = std::min(size-1, int(std::floor(std::max({a.y,b.y,c.y}))));
    for (int y=ymin; y<=ymax; ++y) {
        for (int x=xmin; x<=xmax; ++x) {
            const double wb = ((x-a.x)*(c.y-a.y)-(y-a.y)*(c.x-a.x))/area;
            const double wc = ((b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x))/area;
            const double wa = 1-wb-wc;
            if (std::min({wa,wb,wc}) < 0) continue;
            const double inv = wa/a.z+wb/b.z+wc/c.z;
            const int pixel = y*size+x;
            if (1/inv >= depth[pixel]) continue;
            depth[pixel] = 1/inv;
            gray[pixel] = (wa*a.gray/a.z+wb*b.gray/b.z+wc*c.gray/c.z)/inv;
            labels[pixel] = part;
        }
    }
}
}

extern "C" void render_face(const double *vertices, const int32_t *faces,
                            const uint8_t *parts, int count, int size,
                            float *depth, float *gray, uint8_t *labels) {
    constexpr double near = .02;
    for (int face=0; face<count; ++face) {
        std::array<Vertex,3> source;
        for (int i=0; i<3; ++i) {
            const double *p = vertices+4*faces[3*face+i];
            source[i] = {p[0],p[1],p[2],p[3]};
        }
        std::array<Vertex,4> clipped;
        int n = 0;
        for (int i=0; i<3; ++i) {
            const Vertex a = source[i], b = source[(i+1)%3];
            if (a.z>=near) clipped[n++] = a;
            if ((a.z>=near)!=(b.z>=near))
                clipped[n++] = mix(a,b,(near-a.z)/(b.z-a.z));
        }
        for (int i=1; i+1<n; ++i)
            triangle(clipped[0],clipped[i],clipped[i+1],parts[face],size,depth,gray,labels);
    }
}
