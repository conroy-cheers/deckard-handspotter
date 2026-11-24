#include "handspotter_prepare.h"
#include <algorithm>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

void require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}
} // namespace

extern "C" const char *fhg_error(void) {return last_error.c_str();}

extern "C" int fhg_prepare_handspotter(const fhp_plane source[4],const uint8_t *support[4],uint8_t output[81920]) {
  try {
    require(source && support && output,"Missing Handspotter image argument");
    constexpr uint32_t width=160,height=128;
    for (int camera=0;camera<4;++camera) {
      const auto &p=source[camera];
      require(p.data && support[camera] && p.width>=width && p.height>=height &&
              p.width<=4096 && p.height<=4096 && p.stride>=p.width &&
              p.bytes>=size_t(p.stride)*(p.height-1)+p.width,"Invalid Handspotter image plane");
      const uint32_t gx=std::gcd(p.width,width),gy=std::gcd(p.height,height);
      const uint32_t nx=p.width/gx,dx=width/gx,ny=p.height/gy,dy=height/gy;
      const uint32_t den=nx*ny;
      for (uint32_t y=0;y<height;++y) for (uint32_t x=0;x<width;++x) {
        const uint32_t top=y*ny,bottom=top+ny,left=x*nx,right=left+nx;
        uint32_t sum=0;
        for (uint32_t sy=top/dy;sy<(bottom+dy-1)/dy;++sy) {
          const uint32_t wy=std::min(bottom,(sy+1)*dy)-std::max(top,sy*dy);
          const uint8_t *pixels=p.data+size_t(p.stride)*sy;
          const uint8_t *mask=support[camera]+size_t(p.width)*sy;
          uint32_t row=0;
          for (uint32_t sx=left/dx;sx<(right+dx-1)/dx;++sx) {
            const uint32_t wx=std::min(right,(sx+1)*dx)-std::max(left,sx*dx);
            row+=uint32_t(pixels[sx])*mask[sx]*wx;
          }
          sum+=row*wy;
        }
        const uint32_t code=sum/den,twice_remainder=2*(sum%den);
        *output++=uint8_t(code+(twice_remainder>den || (twice_remainder==den && code%2)));
      }
    }
    last_error.clear();return 0;
  } catch (const std::exception &e) {last_error=e.what();return -1;}
  catch (...) {last_error="Native Handspotter image preparation failed";return -1;}
}
