
#ifndef U1180779_CAST_RAYS_H
#define U1180779_CAST_RAYS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "general.hpp"
#include "unifiedObjects.hpp"

struct castRaysData
{
    int z = 0; // camera z coordinate
    int width = 0;
    int height = 0;
    cudaSurfaceObject_t surfaceObject;

    unifiedObjects data;
};

__global__ void castRaysKernel(castRaysData data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel y

    if (x >= data.width || y >= data.height)
        return;

    // ray origin
    float3 o;
    o.x = x;
    o.y = y;
    o.z = data.z;

    // find closes sphere (or light) by z coordinate
    unifiedObject closest;
    closest.x = FLT_MAX / 2.f;
    closest.y = FLT_MAX / 2.f;
    closest.z = FLT_MAX / 2.f;
    closest.r = 1.f;

    for (int i = 0; i < data.data.count; ++i) {
        __syncthreads();
        // check z (if closer by z coordinate)
        if (!(data.data.z[i] < closest.z)) // (!(data.sData.z[i] - o.z < closest.z - o.z))
            continue;

        //check x and y
        float dist2 = (data.data.x[i] - o.x) * (data.data.x[i] - o.x) + (data.data.y[i] - o.y) * (data.data.y[i] - o.y);
        if (dist2 > data.data.r[i] * data.data.r[i])
            continue;

        closest = data.data(i);
    }

    // if found no sphere return
    if (closest.x == FLT_MAX / 2.f)
    {
        uchar4 notFoundWriteData;
        notFoundWriteData.x = 255;
        notFoundWriteData.y = 255;
        notFoundWriteData.z = 255;
        notFoundWriteData.w = 255;
        surf2Dwrite(notFoundWriteData, data.surfaceObject, 4 * x, y);
        return;
    }

    // get point on the sphere
    //float3 p;
    //p.x = o.x;
    //p.y = o.y;

    //float x2 = x - closest.x;
    //x2 *= x2;
    //float y2 = y - closest.y;
    //y2 *= y2;
    //float z2 = o.z - closest.z;
    //
    //// float a = 1.f;
    //float b = 2 * z2;
    //float c = z2 * z2 + x2 + y2 - closest.r * closest.r;
    //
    //float delta = (b * b - 4 * c) / 2;
    //float i = (-b + sqrt(delta)) / 2; // +- sqrt(delta)

    //p.z = o.z + i;

    // get normal vector
    //float3 N;
    //N.x = p.x - closest.x;
    //N.y = p.y - closest.y;
    //N.z = p.z - closest.z;

    //float t = (float)norm3d(N.x, N.y, N.z);
    //N.x /= t;
    //N.y /= t;
    //N.z /= t;
    

    uchar4 writeData;
    writeData.x = closest.color.x;
    writeData.y = closest.color.y;
    writeData.z = closest.color.z;
    writeData.w = 255;
    surf2Dwrite(writeData, data.surfaceObject, 4 * x, y);
}

#endif


