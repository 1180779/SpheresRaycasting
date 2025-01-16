
#ifndef U1180779_TRANSFORM_SCENE_H
#define U1180779_TRANSFORM_SCENE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lbvhConcrete.cuh"
#include "unifiedObject.hpp"
#include "mat4.cuh"
#include "lights.hpp"

struct transformData 
{
    mat4 t;
    int count;
};

/* kernels run during mouse callback (mouse movement) */

__global__ void callbackAllKernel(transformData data, const bvhDevice ptrs, lights lights);
__global__ void callbackLightsKernel(transformData data, const bvhDevice ptrs, lights lights);


#endif