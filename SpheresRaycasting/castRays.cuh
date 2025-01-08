
#ifndef U1180779_CAST_RAYS_H
#define U1180779_CAST_RAYS_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "spheresData.hpp"

struct castReysData
{
    int width;
    int height;
    cudaSurfaceObject_t surfaceObject;

    spheresData data;
};

__global__ void castReysKernel(castReysData data)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int x = y / data.width;
    y = y % data.width;

    if (x > data.width || y > data.height)
        return;

    surf2Dwrite(0, data.surfaceObject, 0, 0);
}

#endif
