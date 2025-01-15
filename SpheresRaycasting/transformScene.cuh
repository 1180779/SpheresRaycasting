
#ifndef U1180779_TRANSFORM_SCENE_H
#define U1180779_TRANSFORM_SCENE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lbvhConcrete.cuh"
#include "unifiedObjects.hpp"
#include "mat4.cuh"
#include "lights.hpp"

struct transformData 
{
    mat4 t;
    int count;
};

__global__ void transformSceneKernel(transformData data, const bvhDevice ptrs, lights lights)
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


#endif