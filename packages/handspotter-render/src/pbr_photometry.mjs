// All image coordinates here use sensor rows (row zero is the top).
import * as THREE from 'three';

// Each readback holds two hand rectangles and pixel counts, followed by the
// Handspotter input tensor and, when requested, native RGBA pixels.
const SENSOR_HEADER_BYTES=64,TENSOR_WIDTH=160,TENSOR_HEIGHT=128;

const reflectPixel = `
ivec2 reflectPixel(ivec2 p,ivec2 size){
  ivec2 period=max(2*(size-1),ivec2(1));
  ivec2 folded=abs(p)%period;
  return min(folded,period-folded);
}
`;

// Exact overlap of source and destination pixel footprints, including stereo
// sensors with a non-integral resize ratio or a centered crop.
const area = `
uniform sampler2D image;
uniform ivec4 crop;
uniform ivec2 destSize;
vec2 average(ivec2 pixel){
  ivec2 lo=pixel*crop.zw,hi=lo+crop.zw;
  ivec2 first=lo/destSize,end=(hi+destSize-1)/destSize;
  vec2 sum=vec2(0.);
  for(int y=first.y;y<end.y;++y) for(int x=first.x;x<end.x;++x){
    ivec2 pos=ivec2(x,y);
    ivec2 overlap=min(hi,(pos+1)*destSize)-max(lo,pos*destSize);
    vec2 weight=vec2(overlap)/vec2(destSize);
    sum+=texelFetch(image,crop.xy+pos,0).rg*weight.x*weight.y;
  }
  vec2 scale=vec2(crop.zw)/vec2(destSize);
  return sum/(scale.x*scale.y);
}
`;
const shrink=area+`
out vec4 result;
void main(){result=vec4(average(ivec2(gl_FragCoord.xy)),0.,0.);}
`;
const gaussian=reflectPixel+`
uniform sampler2D image;
uniform ivec2 axis;
uniform int radius;
uniform float sigma;
out vec4 result;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy),size=textureSize(image,0);
  vec4 center=texelFetch(image,p,0);
  vec2 sum=center.rg;
  float total=1.;
  for(int i=1;i<=radius;++i){
    float d=float(i)/sigma,weight=exp(-.5*d*d);
    sum+=weight*(texelFetch(image,reflectPixel(p+i*axis,size),0).rg+
                 texelFetch(image,reflectPixel(p-i*axis,size),0).rg);
    total+=2.*weight;
  }
  result=vec4(sum/total,center.ba);
}
`;
const hazeShader=`
uniform sampler2D image,scatter;
uniform vec2 center,spread;
uniform float haze,smudge;
out vec4 result;
float glare(ivec2 p){
  vec2 sampleValue=texelFetch(scatter,clamp(p,ivec2(0),textureSize(scatter,0)-1),0).rg;
  return sampleValue.x/max(sampleValue.y,.10);
}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 linear=texelFetch(image,p,0);
  vec2 size=vec2(textureSize(image,0));
  vec2 uv=gl_FragCoord.xy/size*vec2(textureSize(scatter,0))-.5;
  ivec2 base=ivec2(floor(uv));
  vec2 f=fract(uv);
  float indirect=mix(mix(glare(base),glare(base+ivec2(1,0)),f.x),
                     mix(glare(base+ivec2(0,1)),glare(base+ivec2(1)),f.x),f.y);
  vec2 offset=(2.*vec2(p)/max(size-1.,vec2(1.))-1.-center)/spread;
  float opacity=haze*(.58+smudge*exp(-.5*dot(offset,offset)));
  result=vec4(mix(linear.r,indirect,opacity),linear.gba);
}
`;
const response=`
uniform sampler2D image;
uniform uint seed;
uniform bool normalizedBlur;
out vec4 result;
uint pcg(uint inputValue){
  uint state=inputValue*747796405u+2891336453u;
  uint word=((state>>((state>>28u)+4u))^state)*277803737u;
  return (word>>22u)^word;
}
float uniformOpen(uint x){return (float(x>>9u)+.5)/8388608.;}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 linear=texelFetch(image,p,0);
  float signal=linear.r/(normalizedBlur ? max(linear.g,1e-6) : 1.);
  uint counter=uint(p.y)*uint(textureSize(image,0).x)+uint(p.x);
  uint key=pcg(seed+counter);
  float noise=sqrt(-2.*log(uniformOpen(key)))*cos(6.28318530718*uniformOpen(pcg(key)));
  signal=max(signal+noise*sqrt(.000006+.00011*signal),0.);
  signal=clamp(signal/(1.+.32*signal),0.,1.);
  float gray=signal<=.0031308 ? 12.92*signal : 1.055*pow(signal,1./2.4)-.055;
  result=vec4(roundEven(gray*255.)/255.,linear.b/255.,linear.a,1.);
}
`;
const finishNative=reflectPixel+`
uniform sampler2D image;
uniform float neighbor;
out vec4 result;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy),size=textureSize(image,0);
  vec4 center=texelFetch(image,p,0);
  float sum=0.;
  for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x){
    float weight=(x==0 ? 1.-2.*neighbor : neighbor)*(y==0 ? 1.-2.*neighbor : neighbor);
    sum+=weight*texelFetch(image,reflectPixel(p+ivec2(x,y),size),0).r;
  }
  result=vec4(center.b*roundEven(sum*255.)/255.,center.g,0.,1.);
}
`;
const tensor=area+`
int gray(ivec2 p){return int(round(texelFetch(image,crop.xy+p,0).r*255.));}
// OpenCV's byte INTER_AREA enlargement uses two 11-bit interpolation
// coefficients and a staged fixed-point horizontal/vertical reduction.
float enlarge(ivec2 p){
  ivec2 base=p*crop.zw/destSize,other=min(base+1,crop.zw-1);
  ivec2 remainder=max((p+1)*crop.zw-(base+1)*destSize,ivec2(0))%crop.zw;
  ivec2 weight=ivec2(roundEven(vec2(remainder)*2048./vec2(crop.zw)));
  int top=((2048-weight.x)*gray(base)+weight.x*gray(ivec2(other.x,base.y)))>>4;
  int bottom=((2048-weight.x)*gray(ivec2(base.x,other.y))+weight.x*gray(other))>>4;
  return float(((((2048-weight.y)*top)>>16)+((weight.y*bottom)>>16)+2)>>2);
}
float fractionalArea(ivec2 p){
  ivec2 lo=p*crop.zw,hi=lo+crop.zw;
  ivec2 first=lo/destSize,end=(hi+destSize-1)/destSize;
  // Keep the fractional numerator exact at byte-rounding boundaries.
  // Carry whole bytes per row so large native sensors fit in 32 bits.
  uint denominator=uint(crop.z*crop.w),whole=0u,remainder=0u;
  for(int y=first.y;y<end.y;++y){
    uint row=0u;
    for(int x=first.x;x<end.x;++x){
      int overlap=min(hi.x,(x+1)*destSize.x)-max(lo.x,x*destSize.x);
      row+=uint(gray(ivec2(x,y))*overlap);
    }
    int overlap=min(hi.y,(y+1)*destSize.y)-max(lo.y,y*destSize.y);
    remainder+=row*uint(overlap);
    whole+=remainder/denominator;
    remainder%=denominator;
  }
  uint twice=2u*remainder;
  return float(whole+uint(twice>denominator || (twice==denominator && (whole&1u)!=0u)));
}
out vec4 result;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy)*ivec2(4,1);
  vec4 lanes;
  ivec2 scale=crop.zw/destSize;
  for(int lane=0;lane<4;++lane){
    ivec2 dest=p+ivec2(lane,0);
    if(all(equal(scale*destSize,crop.zw))){
      ivec2 base=crop.xy+dest*scale;
      uint sum=0u,area=uint(scale.x*scale.y);
      for(int y=0;y<scale.y;++y) for(int x=0;x<scale.x;++x)
        sum+=uint(round(255.*texelFetch(image,base+ivec2(x,y),0).r));
      lanes[lane]=float((sum+area/2u)/area)/255.;
    } else if(any(lessThan(crop.zw,destSize))){
      lanes[lane]=enlarge(dest)/255.;
    } else {
      lanes[lane]=fractionalArea(dest)/255.;
    }
  }
  result=lanes;
}
`;

// Reduce the full-sensor rectangle for each hand. Preview and tensor-only
// jobs read these 64 bytes instead of a full-resolution mask.
const rows=`
uniform sampler2D image;
out vec4 result;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  int label=p.x+1,end=textureSize(image,0).x;
  int lo=end,hi=0,count=0;
  for(int x=0;x<end;++x){
    if(int(round(texelFetch(image,ivec2(x,p.y),0).g*255.))==label){
      lo=min(lo,x);hi=x+1;++count;
    }
  }
  result=vec4(lo,hi,count,0.);
}
`;
const boxes=`
uniform sampler2D image;
out vec4 result;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 rect=vec4(1e20,1e20,0.,0.);
  float count=0.;
  for(int y=0;y<textureSize(image,0).y;++y){
    vec3 row=texelFetch(image,ivec2(p.x,y),0).rgb;
    if(row.z>0.){
      rect.xy=min(rect.xy,vec2(row.x,y));
      rect.zw=max(rect.zw,vec2(row.y,y+1));
      count+=row.z;
    }
  }
  result=p.y==0 && count>0. ? rect : vec4(count,0.,0.,0.);
}
`;

export function photometry(renderer,material,target,pass){
  const down=material(shrink,{image:null,crop:null,destSize:null});
  const blur=material(gaussian,{image:null,axis:null,radius:0,sigma:1});
  const smudge=material(hazeShader,{image:null,scatter:null,center:null,spread:null,haze:0,smudge:0});
  const code=material(response,{image:null,seed:0,normalizedBlur:false});
  const smooth=material(finishNative,{image:null,neighbor:0});
  const resample=material(tensor,{image:null,crop:null,destSize:null});
  const scan=material(rows,{image:null});
  const bounds=material(boxes,{image:null});
  const glare=target(80,64),glareTmp=target(80,64);
  const temp=target(1,1),filtered=target(1,1);
  const coded=target(1,1,THREE.UnsignedByteType),native=target(1,1,THREE.UnsignedByteType);
  const packed=target(40,TENSOR_HEIGHT,THREE.UnsignedByteType),rowStats=target(2,1),summary=target(2,2);
  function gaussianBlur(input,tmp,dest,sigma){
    const radius=Math.max(1,Math.floor(sigma*4+.5));
    pass(blur,tmp,{image:input.texture,axis:[1,0],radius,sigma});
    pass(blur,dest,{image:tmp.texture,axis:[0,1],radius,sigma});
  }
  return {
    read(linear,settings,includeNative){
      const {width:w,height:h}=linear;
      const full=[0,0,w,h];
      const lens=settings.haze;
      let input=linear;
      if(lens?.haze){
        pass(down,glare,{image:input.texture,crop:full,destSize:[80,64]});
        gaussianBlur(glare,glareTmp,glare,6);
        filtered.setSize(w,h);
        pass(smudge,filtered,{image:input.texture,scatter:glare.texture,center:lens.center,
          spread:lens.width,haze:lens.haze,smudge:lens.smudge});
        input=filtered;
      }
      const sigma=settings.optical_sigma*w/1280;
      if(sigma>0.){
        temp.setSize(w,h);filtered.setSize(w,h);
        gaussianBlur(input,temp,filtered,sigma);
        input=filtered;
      }
      coded.setSize(w,h);native.setSize(w,h);
      pass(code,coded,{image:input.texture,seed:settings.noise_seed,normalizedBlur:sigma>0.});
      const edge=Math.exp(-.5/settings.filter_sigma**2);
      pass(smooth,native,{image:coded.texture,neighbor:edge/(1+2*edge)});
      pass(resample,packed,{image:native.texture,crop:full,destSize:[TENSOR_WIDTH,TENSOR_HEIGHT]});
      rowStats.setSize(2,h);
      pass(scan,rowStats,{image:native.texture});
      pass(bounds,summary,{image:rowStats.texture});
      const end=SENSOR_HEADER_BYTES+TENSOR_WIDTH*TENSOR_HEIGHT;
      const bytes=new Uint8Array(end+(includeNative ? 4*w*h : 0));
      renderer.readRenderTargetPixels(summary,0,0,2,2,new Float32Array(bytes.buffer,0,SENSOR_HEADER_BYTES/4));
      renderer.readRenderTargetPixels(packed,0,0,TENSOR_WIDTH/4,TENSOR_HEIGHT,bytes.subarray(SENSOR_HEADER_BYTES,end));
      if(includeNative) renderer.readRenderTargetPixels(native,0,0,w,h,bytes.subarray(end));
      return bytes;
    },
    dispose(){
      for(const item of [down,blur,smudge,code,smooth,resample,scan,bounds,glare,glareTmp,temp,filtered,
        coded,native,packed,rowStats,summary]) item.dispose();
    },
  };
}
