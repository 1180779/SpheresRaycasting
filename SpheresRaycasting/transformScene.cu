
#include "transformScene.hpp"

__global__ void callbackAllKernel(transformData data, const bvhDevice ptrs, lights lights)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // object to transform

    //const auto ptrs = bvh.get_device_repr();
    if (i >= ptrs.num_objects)
        return;
    vec4 r(ptrs.objects[i].x, ptrs.objects[i].y, ptrs.objects[i].z, ptrs.objects[i].w);
    r = data.t * r;
    ptrs.objects[i].x = r(0);
    ptrs.objects[i].y = r(1);
    ptrs.objects[i].z = r(2);
    ptrs.objects[i].w = r(3);

    // transform copy of lights
    if (i >= lights.count)
        return;
    r = vec4(lights.x[i], lights.y[i], lights.z[i], lights.w[i]);
    r = data.t * r;
    lights.x[i] = r(0);
    lights.y[i] = r(1);
    lights.z[i] = r(2);
    lights.w[i] = r(3);
}

__global__ void callbackLightsKernel(transformData data, const bvhDevice ptrs, lights lights)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // object to transform

    //const auto ptrs = bvh.get_device_repr();
    if (i >= ptrs.num_objects)
        return;

    // transform only the lights
    if (ptrs.objects[i].type == types::lightSource)
    {
        vec4 r(ptrs.objects[i].x, ptrs.objects[i].y, ptrs.objects[i].z, ptrs.objects[i].w);
        r = data.t * r;
        ptrs.objects[i].x = r(0);
        ptrs.objects[i].y = r(1);
        ptrs.objects[i].z = r(2);
        ptrs.objects[i].w = r(3);
    }

    // transform copy of lights
    if (i >= lights.count)
        return;

    vec4 r = vec4(lights.x[i], lights.y[i], lights.z[i], lights.w[i]);
    r = data.t * r;
    lights.x[i] = r(0);
    lights.y[i] = r(1);
    lights.z[i] = r(2);
    lights.w[i] = r(3);
}
