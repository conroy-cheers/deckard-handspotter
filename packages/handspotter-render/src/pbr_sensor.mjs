// Calibrated sensor-space passes. Texture row zero is the top sensor row; the
// cubemap render targets use GL rows, so only the face fetch flips Y.
import * as THREE from 'three';
import { photometry } from '/pbr_photometry.mjs';

const vertex = `
in vec3 position;
void main(){gl_Position=vec4(position,1.);}
`;
const header = `precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
`;

// The projection and its Jacobian are shared by the inverse solve and the
// motion pass. Angular coordinates have magnitude equal to incidence in radians.
const optics = `
struct Lens { vec2 f,c; float k[12]; };
uniform Lens sourceLens, frameLens;
uniform float fraction, angularLimit, norm;
uniform vec2 center, shift, tangential;
vec2 endpoint(Lens lens,vec2 a,out mat2 jac){
  float r=dot(a,a), poly=0., derivative=0.;
  for(int i=5;i>=0;--i){
    derivative=derivative*r+poly;
    poly=poly*r+lens.k[i];
  }
  float radial=1.+r*poly;
  mat2 radialJac=mat2(radial)+2.*(poly+r*derivative)*outerProduct(a,a);
  vec2 uv=a*radial;
  float s=dot(uv,uv);
  vec2 p=vec2(lens.k[6],lens.k[7]);
  vec2 prism=vec2(lens.k[8],lens.k[10]), quartic=vec2(lens.k[9],lens.k[11]);
  float scale=1.+2.*dot(p,uv);
  mat2 warp=mat2(scale)+2.*(outerProduct(uv,p)+outerProduct(p+prism+2.*quartic*s,uv));
  jac=mat2(lens.f.x,0.,0.,lens.f.y)*warp*radialJac;
  return lens.c+lens.f*(uv*scale+(p+prism+quartic*s)*s);
}
vec2 projectAngles(vec2 a,out mat2 jac){
  vec2 pixel=endpoint(sourceLens,a,jac);
  if(fraction>0.){
    mat2 other;
    vec2 frame=endpoint(frameLens,a,other);
    pixel=mix(pixel,frame,fraction);
    jac=jac*(1.-fraction)+other*fraction;
  }
  vec2 uv=(pixel-center)/norm;
  float scale=1.+2.*dot(tangential,uv);
  jac=(mat2(scale)+2.*(outerProduct(uv,tangential)+outerProduct(tangential,uv)))*jac;
  return pixel+shift+norm*(uv*(scale-1.)+tangential*dot(uv,uv));
}
`;
const inverseProjection = optics+`
out vec4 result;
void main(){
  vec2 pixel=gl_FragCoord.xy-.5;
  vec2 focal=mix(sourceLens.f,frameLens.f,fraction);
  vec2 offset=(pixel-mix(sourceLens.c,frameLens.c,fraction)-shift)/focal;
  // Bracket the radial solution first, including the outer edge of lenses
  // whose radial derivative approaches zero.
  vec2 dir=offset/max(length(offset),1e-12);
  float lo=0.,hi=angularLimit;
  mat2 jac;
  for(int i=0;i<9;++i){
    float mid=(lo+hi)*.5;
    vec2 estimate=projectAngles(dir*mid,jac);
    if(dot((estimate-pixel)/focal,dir)<0.) lo=mid; else hi=mid;
  }
  vec2 a=dir*((lo+hi)*.5);
  for(int i=0;i<16;++i){
    vec2 err=projectAngles(a,jac)-pixel;
    if(max(abs(err.x),abs(err.y))<.00008) break;
    vec2 step=inverse(jac)*err;
    a-=step/max(1.,length(step)/.1);
    a*=min(1.,(angularLimit+.0001)/max(length(a),1e-12));
  }
  vec2 err=abs(projectAngles(a,jac)-pixel);
  bool valid=length(a)<=angularLimit && !any(isnan(a));
  float residual=valid ? max(err.x,err.y) : 0.;
  if(any(isnan(a)) || (valid && determinant(jac)<=0.)) residual=1e10;
  result=vec4(a,valid ? 1. : 0.,residual);
}
`;

const projectSensor = optics+`
uniform sampler2D rays;
uniform sampler2DArray beauty, labels;
uniform mat3 eyeToWorld;
uniform vec3 omega;
uniform bool motion;
uniform int faceSize;
out vec4 result;
ivec3 facePixel(vec3 d){
  vec3 local=d;
  int face=0;
  if(d.x>local.z){face=1;local=vec3(-d.z,d.y,d.x);}
  if(-d.x>local.z){face=2;local=vec3(d.z,d.y,-d.x);}
  if(-d.y>local.z){face=3;local=vec3(d.x,d.z,-d.y);}
  if(d.y>local.z){face=4;local=vec3(d.x,-d.z,d.y);}
  ivec2 xy=clamp(ivec2(roundEven((local.xy/local.z+1.)*.5*float(faceSize)-.5)),ivec2(0),ivec2(faceSize-1));
  return ivec3(xy.x,faceSize-1-xy.y,face);
}
void main(){
  vec4 optical=texelFetch(rays,ivec2(gl_FragCoord.xy),0);
  if(optical.z==0.){result=vec4(0.);return;}
  vec2 a=optical.xy;
  float theta=length(a), s=sin(theta), c=cos(theta);
  vec2 dir=theta>1e-7 ? a/theta : vec2(1.,0.);
  vec3 ray=vec3(dir*s,c);
  ivec3 src=facePixel(ray);
  vec4 label=texelFetch(labels,src,0);
  float luma=round(clamp(dot(texelFetch(beauty,src,0).rgb,vec3(.2126,.7152,.0722))*4096.,0.,65535.));
  int part=motion ? max(int(label.a)-1,0)%4 : int(round(label.r*255.));
  int metadata=(1+int(round(c*c*16382.)))*4+((part==1 || part==2) ? part : 0);
  vec2 flow=vec2(0.);
  if(motion){
    vec3 v=label.a==0. ? -cross(omega,eyeToWorld*ray) : label.xyz;
    v=transpose(eyeToWorld)*v;
    vec2 tangent=vec2(-dir.y,dir.x);
    vec2 angular=dir*(c*dot(dir,v.xy)-s*v.z)+tangent*dot(tangent,v.xy)*(theta<1e-6 ? 1. : theta/s);
    mat2 jac;
    projectAngles(a,jac);
    flow=jac*angular;
  }
  result=vec4(luma,label.a>4. && motion ? -float(metadata) : float(metadata),flow);
}
`;

// Scatter silhouette velocity into the pixels swept during the shutter.
// Each moving pixel rasterizes only the bounding rectangle of its path.
// The depth test selects the nearest source, matching the original filter.
const sweepVertex = `
uniform sampler2D sensor;
flat out ivec2 origin;
flat out vec2 flow;
void main(){
  ivec2 size=textureSize(sensor,0);
  origin=ivec2(gl_InstanceID%size.x,gl_InstanceID/size.x);
  vec4 sampleValue=texelFetch(sensor,origin,0);
  flow=sampleValue.zw;
  if(sampleValue.y>=0. || max(abs(flow.x),abs(flow.y))<1.){gl_Position=vec4(2.,2.,0.,1.);return;}
  vec2 corners[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
  vec2 p=vec2(origin)+.5+corners[gl_VertexID]*(abs(flow)*.5+1.);
  gl_Position=vec4(p/vec2(size)*2.-1.,0.,1.);
}
`;
const sweepFragment = `
uniform sampler2D sensor;
flat in ivec2 origin;
flat in vec2 flow;
out vec4 result;
void main(){
  ivec2 target=ivec2(gl_FragCoord.xy);
  if(texelFetch(sensor,target,0).y<=0.) discard;
  vec2 delta=vec2(target-origin);
  float direction=dot(delta,flow)>0. ? .5 : -.5;
  int steps=min(32,int(ceil(max(abs(flow.x),abs(flow.y))*.5)));
  for(int j=steps;j>0;--j){
    ivec2 p=origin+ivec2(floor(flow*(direction*float(j)/float(steps))+.5));
    if(any(lessThan(p,ivec2(0))) || any(greaterThanEqual(p,textureSize(sensor,0)))) continue;
    if(texelFetch(sensor,p,0).y<0.) break;
    if(all(equal(p,target))){
      gl_FragDepth=1.-1./(1.+dot(delta,delta)/256.);
      result=vec4(flow,1.,0.);return;
    }
  }
  discard;
}
`;
const expose = `
uniform sampler2D sensor, swept;
uniform bool motion, arms;
uniform float exposure;
out vec4 result;
float interpolateSignal(vec2 p,float fallback){
  ivec2 start=ivec2(floor(p));
  vec2 frac=fract(p);
  float total=0.,coverage=0.;
  for(int y=0;y<=1;++y) for(int x=0;x<=1;++x){
    ivec2 q=start+ivec2(x,y);
    if(any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,textureSize(sensor,0)))) continue;
    vec4 sampleValue=texelFetch(sensor,q,0);
    if(sampleValue.y==0.) continue;
    float w=(x==1 ? frac.x : 1.-frac.x)*(y==1 ? frac.y : 1.-frac.y);
    total+=sampleValue.x*w;coverage+=w;
  }
  return coverage>0. ? total/coverage : fallback;
}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 sampleValue=texelFetch(sensor,p,0);
  float signal=sampleValue.x;
  if(motion && sampleValue.y!=0.){
    vec2 v=sampleValue.zw;
    if(arms){
      vec4 extension=texelFetch(swept,p,0);
      if(extension.z>0.) v=extension.xy;
    }
    float speed=length(v);
    if(speed>=.35){
      int taps=min(6,int(ceil(speed*.5)));
      for(int j=1;j<=taps;++j){
        vec2 step=v*(float(j)/float(taps*2+1));
        signal+=interpolateSignal(vec2(p)+step,sampleValue.x)+interpolateSignal(vec2(p)-step,sampleValue.x);
      }
      signal/=float(taps*2+1);
    }
  }
  int metadata=int(abs(sampleValue.y));
  float valid=metadata==0 ? 0. : 1.;
  float vignette=.78+.22*float((metadata>>2)-1)/16382.;
  result=vec4(round(signal)*exposure*vignette/4096.,valid,float(metadata&3),valid);
}
`;

const reduce = `
uniform sampler2D inputMap;
out vec4 result;
void main(){
  ivec2 base=ivec2(gl_FragCoord.xy)*4,size=textureSize(inputMap,0);
  vec2 stats=vec2(0.);
  for(int y=0;y<4;++y) for(int x=0;x<4;++x){
    ivec2 p=base+ivec2(x,y);
    if(any(greaterThanEqual(p,size))) continue;
    vec2 pair=texelFetch(inputMap,p,0).zw;
    stats=vec2(stats.x+pair.x,max(stats.y,pair.y));
  }
  result=vec4(0.,0.,stats);
}
`;

function material(fragment, uniforms, vert=vertex, depth=false){
  return new THREE.RawShaderMaterial({glslVersion:THREE.GLSL3, vertexShader:header+vert,fragmentShader:header+fragment,
    uniforms:Object.fromEntries(Object.entries(uniforms).map(([k,value])=>[k,{value}])),
    depthTest:depth,depthWrite:depth,depthFunc:THREE.LessDepth,blending:THREE.NoBlending});
}
function target(w,h,type=THREE.FloatType,depthBuffer=false){
  return new THREE.WebGLRenderTarget(w,h,{type,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer});
}
function lensUniforms(lens){
  return {f:new THREE.Vector2(...lens.f),c:new THREE.Vector2(...lens.c),k:lens.coefficients};
}

export function sensorCapture(renderer){
  const camera=new THREE.Camera();
  const triangle=new THREE.BufferGeometry();
  triangle.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,3,-1,0,-1,3,0],3));
  const instances=new THREE.InstancedBufferGeometry();
  instances.setAttribute('position',triangle.attributes.position);
  const mesh=new THREE.Mesh(triangle);
  mesh.frustumCulled=false;
  const scene=new THREE.Scene();
  scene.add(mesh);
  const opticalUniforms={sourceLens:null,frameLens:null,fraction:0,angularLimit:0,norm:1,center:null,shift:null,tangential:null};
  const inversion=material(inverseProjection,opticalUniforms);
  const project=material(projectSensor,{...opticalUniforms,rays:null,beauty:null,labels:null,
    eyeToWorld:null,omega:null,motion:false,faceSize:0});
  const sweep=material(sweepFragment,{sensor:null},sweepVertex,true);
  const shutter=material(expose,{sensor:null,swept:null,motion:false,arms:false,exposure:1});
  const finish=photometry(renderer,material,target,pass);
  const reduction=material(reduce,{inputMap:null});
  const sensor=target(1,1),swept=target(1,1,THREE.FloatType,true),linear=target(1,1);
  const maps=[];
  function pass(shader,dest,values,geometry=triangle){
    for(const [key,value] of Object.entries(values)) shader.uniforms[key].value=value;
    mesh.material=shader;
    mesh.geometry=geometry;
    renderer.setRenderTarget(dest);
    renderer.render(scene,camera);
  }
  function validate(map,limit){
    let input=map;
    do {
      const next=target(Math.ceil(input.width/4),Math.ceil(input.height/4));
      pass(reduction,next,{inputMap:input.texture});
      if(input!==map) input.dispose();
      input=next;
    } while(input.width>1 || input.height>1);
    const stats=new Float32Array(4);
    try { renderer.readRenderTargetPixels(input,0,0,1,1,stats); }
    finally { input.dispose(); }
    if(!stats[2] || !(stats[3]<=.01)) throw new Error('GPU lens inversion failed: '+stats);
    return {max_roundtrip_pixels:stats[3],valid_fraction:stats[2]/(map.width*map.height),max_rendered_incidence_deg:limit*180/Math.PI};
  }
  return {
    read(job,frame,eye,exposure,beauty,labels,motion,angular,validation){
      const spec=job.optics[eye], [w,h]=job.sensor_sizes[eye];
      const scale=w/1280, lens=spec.decenter;
      const uniforms={sourceLens:lensUniforms(spec.source),frameLens:lensUniforms(spec.frame),
        fraction:spec.fraction,angularLimit:spec.limit,norm:640*scale,
        center:new THREE.Vector2(...(lens?.center_native_xy_px ?? [0,0])).addScalar(.5).multiplyScalar(scale).subScalar(.5),
        shift:new THREE.Vector2(...(lens?.shift_native_xy_px ?? [0,0])).multiplyScalar(scale),
        tangential:new THREE.Vector2(...(lens?.tangential_p1_p2 ?? [0,0]).slice().reverse())};
      const key=JSON.stringify([w,h,spec]);
      if(!maps[eye]) maps[eye]={map:target(w,h)};
      const cached=maps[eye];
      if(cached.key!==key){
        cached.map.setSize(w,h);
        pass(inversion,cached.map,uniforms);
        const stats=validate(cached.map,spec.limit);
        if(validation) validation[eye]=stats;
        cached.key=key;
      }
      sensor.setSize(w,h);linear.setSize(w,h);
      pass(project,sensor,{...uniforms,rays:cached.map.texture,beauty,labels,motion:!!motion,
        eyeToWorld:new THREE.Matrix3().set(...spec.rotation.flat()),omega:angular,faceSize:job.size});
      if(motion?.arms){
        swept.setSize(w,h);
        instances.instanceCount=w*h;
        renderer.setClearColor(0,0);
        pass(sweep,swept,{sensor:sensor.texture},instances);
      }
      pass(shutter,linear,{sensor:sensor.texture,swept:swept.texture,motion:!!motion,arms:!!motion?.arms,exposure});
      return finish.read(linear,job.sensor_settings[frame][eye],job.native_images);
    },
    dispose(){
      for(const {map} of maps) map.dispose();
      for(const item of [inversion,project,sweep,shutter,finish,reduction,sensor,swept,linear,triangle,instances]) item.dispose();
    },
  };
}
