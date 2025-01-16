
#ifndef U1180779_CAST_RAYS_CUH
#define U1180779_CAST_RAYS_CUH

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "cudaWrappers.hpp"
#include "unifiedObject.hpp"
#include "unifiedObjects.cuh"
#include "lights.hpp"
#include "lbvhConcrete.cuh"
#include "rays.cuh"

/* inline floatN Operations */
/* ------------------------------------------------------------------------------- */

/* operators */

__device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator+(float3 v, float f)
{
    return make_float3(v.x + f, v.y + f, v.z + f);
}

__device__ __forceinline__ float3 operator+(float f, float3 v)
{
    return v + f;
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(float f, float3 v)
{
    return make_float3(f * v.x, f * v.y, f * v.z);
}

__device__ __forceinline__ float3 operator*(float3 v, float f)
{
    return f * v;
}

__device__ __forceinline__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(float3 v, float f)
{
    return make_float3(v.x / f, v.y / f, v.z / f);
}

/* vector math */

__device__ __forceinline__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize(float3 v)
{
    float l = sqrt(dot(v, v));
    return v / l;
}

__device__ __forceinline__ float3 min(float3 a, float f)
{
    return make_float3(fminf(a.x, f), fminf(a.y, f), fminf(a.z, f));
}

__device__ __forceinline__ float3 max(float3 a, float f)
{
    return make_float3(fmaxf(a.x, f), fmaxf(a.y, f), fmaxf(a.z, f));
}

/* kernel to cast rays */
/* ------------------------------------------------------------------------------- */

__global__ void findClosestKernel(const bvhDevice ptrs, int width, int height, unifiedObjects objs, lights lights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel y

    if (x >= width || y >= height)
        return;

    float3 O; // ray origin
    O.x = x;
    O.y = y;
    O.z = 0;

    // find closes sphere (or light) by z coordinate
    unifiedObject closest;
    closest.x = FLT_MAX / 2.f;
    closest.y = FLT_MAX / 2.f;
    closest.z = FLT_MAX / 2.f;
    closest.r = 1.f;

    // move throught the bhv tree
    //auto ptrs = bvh.get_device_repr();

    bvhNodeIdx stack[32]; // local stack
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    while (stack_ptr > 0) {

        int indx = stack[--stack_ptr];
        bvhNode current = ptrs.nodes[indx];

        if (!rayHit(ptrs.aabbs[indx].lower, ptrs.aabbs[indx].upper, O)) // no hit with the box, can skip this subtree
            continue;
        if (current.left_idx == 0xFFFFFFFF) // is leaf
        {
            int i = current.object_idx;
            // check if closer
            
            // check z (if closer by z coordinate)
            if (!(ptrs.objects[i].z < closest.z)) // (!(data.sData.z[i] - o.z < closest.z - o.z))
                continue;

            //check x and y
            float dist2 = (ptrs.objects[i].x - O.x) * (ptrs.objects[i].x - O.x) + (ptrs.objects[i].y - O.y) * (ptrs.objects[i].y - O.y);
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

    objs.set(closest, y * width + x);
}

__global__ void drawColorKernel(unifiedObjects objs, int width, int height, cudaSurfaceObject_t surfaceObject, lights lights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel y

    if (x >= width || y >= height)
        return;
    int i = y * width + x;

    // if found no sphere return
    if (objs.x[i] == FLT_MAX / 2.f)
    {
        uchar4 notFoundWriteData;
        notFoundWriteData.x = (unsigned char)(lights.clearColor.x * 255.0f);
        notFoundWriteData.y = (unsigned char)(lights.clearColor.y * 255.0f);
        notFoundWriteData.z = (unsigned char)(lights.clearColor.z * 255.0f);
        notFoundWriteData.w = (unsigned char)(lights.clearColor.w * 255.0f);
        surf2Dwrite(notFoundWriteData, surfaceObject, 4 * x, y);
        return;
    }

    if (objs.type[i] == types::lightSource) {
        uchar4 writeData;
        writeData.x = objs.colorX[i];
        writeData.y = objs.colorY[i];
        writeData.z = objs.colorZ[i];
        writeData.w = 255;
        surf2Dwrite(writeData, surfaceObject, 4 * x, y);
        return;
    }

    __syncthreads();

    float3 O; // ray origin
    O.x = x;
    O.y = y;
    O.z = 0;

    float3 D; // ray direction
    D.x = 0;
    D.y = 0;
    D.z = 1.0f;

    // get point on the sphere
    float3 C = make_float3(objs.x[i], objs.y[i], objs.z[i]);
    //float a = 1.0f; 
    float b = 2 * dot(D, O - C);
    float c = dot(O - C, O - C) - objs.r[i] * objs.r[i];

    float delta = b * b - 4 * c;
    delta = sqrt(delta);
    float i1 = ((-b - delta) / 2.0f);
    float i2 = ((-b + delta) / 2.0f);
    float t = i1 < i2 ? i1 : i2;

    float3 P = O + t * D;
    float3 N = normalize(P - C);
    float3 V = normalize(O - P);


    float3 objColor = make_float3(objs.colorX[i], objs.colorY[i], objs.colorZ[i]);
    // calculate light
    float3 color = lights.ia * objs.ka[i] * objColor;
    for (int i = 0; i < lights.count; ++i) {
        __syncthreads();

        float3 L = normalize(make_float3(lights.x[i], lights.y[i], lights.z[i]) - C);
        float3 R = normalize(2 * (dot(L, N)) * N - L);

        color = color +
            lights.id[i] * objs.kd[i] * objColor * fmaxf(0.0f, dot(L, N)) +
            lights.is[i] * objs.ks[i] * objColor * __powf(fmaxf(0.0f, dot(R, V)), objs.alpha[i]);
    }

    __syncthreads();

    color = min(color, 1.0f);

    uchar4 writeData;
    writeData.x = (unsigned char)(color.x * 255.0f);
    writeData.y = (unsigned char)(color.y * 255.0f);
    writeData.z = (unsigned char)(color.z * 255.0f);
    writeData.w = 255;
    surf2Dwrite(writeData, surfaceObject, 4 * x, y);
}


#endif


