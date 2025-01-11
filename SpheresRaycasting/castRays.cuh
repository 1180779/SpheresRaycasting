
#ifndef U1180779_CAST_RAYS_H
#define U1180779_CAST_RAYS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "general.hpp"

#include "spheresData.hpp"
#include "lightData.hpp"
#include "spheres.hpp"

struct castRaysData
{
    int z = 0; // camera z coordinate
    int width = 0;
    int height = 0;
    cudaSurfaceObject_t surfaceObject;

    spheresData sData;
    lightData lData;
};

struct castRaysSortTempData
{
    int* keys;
    castRaysData data;
    castRaysData temp;

    void malloc(spheres& data);
    void free();
};

void castRaysSortTempData::malloc(spheres& data)
{
    this->data.sData = data.md_spheres;
    xcudaMalloc(&keys, sizeof(int) * data.hCount());
    temp.sData.dMalloc(data.hCount());
}

void castRaysSortTempData::free()
{
    xcudaFree(keys);
    temp.sData.dFree();
}

__global__ void setKeys(castRaysSortTempData data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data.data.sData.count)
        return;
    data.keys[i] = i;
}

__host__ void sortByZ(castRaysSortTempData data)
{
    thrust::sort_by_key(thrust::device_ptr<int>(data.keys),
        thrust::device_ptr<int>(data.keys + data.data.sData.count),
        thrust::device_ptr<float>(data.data.sData.z));
}

__global__ void copyToTempKernel(castRaysSortTempData data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data.data.sData.count)
        return;
    int index = data.keys[i];

    data.temp.sData.x[index] = data.data.sData.x[i];
    data.temp.sData.y[index] = data.data.sData.y[i];
    data.temp.sData.z[index] = data.data.sData.z[i];
    data.temp.sData.w[index] = data.data.sData.w[i];
    data.temp.sData.r[index] = data.data.sData.r[i];
    data.temp.sData.color[index] = data.data.sData.color[i];
}

__global__ void copyFromTempKernel(castRaysSortTempData data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data.data.sData.count)
        return;

    data.data.sData.x[i] = data.temp.sData.x[i];
    data.data.sData.y[i] = data.temp.sData.y[i];
    data.data.sData.z[i] = data.temp.sData.z[i];
    data.data.sData.w[i] = data.temp.sData.w[i];
    data.data.sData.r[i] = data.temp.sData.r[i];
    data.data.sData.color[i] = data.temp.sData.color[i];
}

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
    sphereData closest;
    closest.x = FLT_MAX / 2.f;
    closest.y = FLT_MAX / 2.f;
    closest.z = FLT_MAX / 2.f;
    closest.r = 1.f;

    for (int i = 0; i < data.sData.count; ++i) {
        __syncthreads();
        // check z (if closer by z coordinate)
        if (!(data.sData.z[i] < closest.z)) // (!(data.sData.z[i] - o.z < closest.z - o.z))
            continue;

        //check x and y
        float dist2 = (data.sData.x[i] - o.x) * (data.sData.x[i] - o.x) + (data.sData.y[i] - o.y) * (data.sData.y[i] - o.y);
        if (dist2 > data.sData.r[i] * data.sData.r[i])
            continue;

        closest.x = data.sData.x[i];
        closest.y = data.sData.y[i];
        closest.z = data.sData.z[i];
        closest.r = data.sData.r[i];
        closest.color = data.sData.color[i];
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


