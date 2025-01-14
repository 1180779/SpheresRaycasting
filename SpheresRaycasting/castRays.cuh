
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
#include "unifiedObjects.cuh"

#include "lbvhConcrete.cuh"

__device__ __forceinline__ bool rayHit(float4 aabbMin, float4 aabbMax, float3 o) 
{
    return aabbMin.x < o.x && o.x < aabbMax.x &&
        aabbMin.y < o.y && o.y < aabbMax.y;
}

//__global__ void castRaysKernel(bvh bvh, int width, int height, cudaSurfaceObject_t surfaceObject)
__global__ void castRaysKernel(const bvhDevice ptrs, int width, int height, cudaSurfaceObject_t surfaceObject)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel y

    if (x >= width || y >= height)
        return;

    // ray origin
    float3 o;
    o.x = x;
    o.y = y;
    o.z = 0;

    // find closes sphere (or light) by z coordinate
    unifiedObject closest;
    closest.x = FLT_MAX / 2.f;
    closest.y = FLT_MAX / 2.f;
    closest.z = FLT_MAX / 2.f;
    closest.r = 1.f;

    // move throught the bhv tree
    //auto ptrs = bvh.get_device_repr();

    bvhNodeIdx stack[64]; // local stack
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    while (stack_ptr > 0) {

        int indx = stack[--stack_ptr];
        bvhNode current = ptrs.nodes[indx];

        if (!rayHit(ptrs.aabbs[indx].lower, ptrs.aabbs[indx].upper, o)) // no hit with the box, can skip this subtree
            continue;
        if (current.left_idx == 0xFFFFFFFF) // is leaf
        {
            int i = current.object_idx;
            // check if closer
            
            // check z (if closer by z coordinate)
            if (!(ptrs.objects[i].z < closest.z)) // (!(data.sData.z[i] - o.z < closest.z - o.z))
                continue;

            //check x and y
            float dist2 = (ptrs.objects[i].x - o.x) * (ptrs.objects[i].x - o.x) + (ptrs.objects[i].y - o.y) * (ptrs.objects[i].y - o.y);
            if (dist2 > ptrs.objects[i].r * ptrs.objects[i].r)
                continue;
            closest = ptrs.objects[i];
        }
        else 
        {
            if (current.left_idx)
                stack[stack_ptr++] = current.left_idx;
            if (current.right_idx) 
                stack[stack_ptr++] = current.right_idx;
        }
    }
    __syncthreads();

    // if found no sphere return
    if (closest.x == FLT_MAX / 2.f)
    {
        uchar4 notFoundWriteData;
        notFoundWriteData.x = 255;
        notFoundWriteData.y = 255;
        notFoundWriteData.z = 255;
        notFoundWriteData.w = 255;
        surf2Dwrite(notFoundWriteData, surfaceObject, 4 * x, y);
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
    surf2Dwrite(writeData, surfaceObject, 4 * x, y);
}

#endif


