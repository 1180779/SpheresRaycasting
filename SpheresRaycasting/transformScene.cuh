
#ifndef U1180779_TRANSFORM_SCENE_H
#define U1180779_TRANSFORM_SCENE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "spheresData.hpp"
#include "lightData.hpp"
#include "mat4.cuh"


struct transformData 
{
    mat4 t;

    spheresData sData;
    lightData lData;
};

__global__ void transformSceneKernel(transformData data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // sphere to transform

    if (i >= data.sData.count)
        return;
    vec4 r(data.sData.x[i], data.sData.y[i], data.sData.z[i], data.sData.w[i]);
    r = data.t * r;
    data.sData.x[i] = r(0);
    data.sData.y[i] = r(1);
    data.sData.z[i] = r(2);
    data.sData.w[i] = r(3);
}


#endif