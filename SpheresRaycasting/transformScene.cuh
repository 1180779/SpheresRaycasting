
#ifndef U1180779_TRANSFORM_SCENE_H
#define U1180779_TRANSFORM_SCENE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "unifiedObjects.hpp"
#include "mat4.cuh"


struct transformData 
{
    mat4 t;

    unifiedObjects data;
};

__global__ void transformSceneKernel(transformData data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // object to transform

    if (i >= data.data.count)
        return;
    vec4 r(data.data.x[i], data.data.y[i], data.data.z[i], data.data.w[i]);
    r = data.t * r;
    data.data.x[i] = r(0);
    data.data.y[i] = r(1);
    data.data.z[i] = r(2);
    data.data.w[i] = r(3);
}


#endif