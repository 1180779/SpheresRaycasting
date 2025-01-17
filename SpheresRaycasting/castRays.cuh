
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
#include "lights.hpp"
#include "lbvhConcrete.cuh"
#include "rays.cuh"

/* inline float3 operations */
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

__global__ void castRaysKernel(const bvhDevice ptrs, 
    int width, int height, 
    float scaleX, float scaleY, /* currently not used (1.0f, 1.0f) */
    cudaSurfaceObject_t surfaceObject, lights lights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel y

    if (x >= width || y >= height)
        return;

    // ray origin
    float3 O;
    O.x = x * scaleX;
    O.y = y * scaleY;
    O.z = 0;

    float3 D;
    D.x = 0;
    D.y = 0;
    D.z = 1.0f;

    /* find closes sphere(or light) by z coordinate */ 
    unifiedObject closest;
    closest.x = FLT_MAX / 2.f;
    closest.y = FLT_MAX / 2.f;
    closest.z = FLT_MAX / 2.f;
    closest.r = 1.f;

    /* move throught the bhv tree */ 
    /* note: shared memory stack appears to be slower */
    bvhNodeIdx stack[32]; 
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    while (stack_ptr > 0) {

        int indx = stack[--stack_ptr];
        bvhNode current = ptrs.nodes[indx];

        if (!rayHit(ptrs.aabbs[indx].lower, ptrs.aabbs[indx].upper, O)) /* no hit with the box, can skip this subtree */ 
            continue;
        if (current.left_idx == 0xFFFFFFFF) /* is leaf */
        {
            int i = current.object_idx;

            /* check if hit is closer */
            
            /* check which spehre is closer by z coordinate of its center 
                * assumes that spheres are of the same radius and are not overlapping */
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

    /* if found no sphere, write background color to texture */
    if (closest.x == FLT_MAX / 2.f)
    {
        uchar4 notFoundWriteData;
        notFoundWriteData.x = (unsigned char)(lights.clearColor.x * 255.0f);
        notFoundWriteData.y = (unsigned char)(lights.clearColor.y * 255.0f);
        notFoundWriteData.z = (unsigned char)(lights.clearColor.z * 255.0f);
        notFoundWriteData.w = (unsigned char)(lights.clearColor.w * 255.0f);
        surf2Dwrite(notFoundWriteData, surfaceObject, 4 * x, y);
        return;
    }

    if (closest.type == types::lightSource) 
    {
        uchar4 writeData;
        writeData.x = (unsigned char)(closest.color.x * 255.0f);
        writeData.y = (unsigned char)(closest.color.y * 255.0f);
        writeData.z = (unsigned char)(closest.color.z * 255.0f);
        writeData.w = 255;
        surf2Dwrite(writeData, surfaceObject, 4 * x, y);
        return;
    }

    /* get point on the sphere */ 
    float3 C = make_float3(closest.x, closest.y, closest.z);
    //float a = 1.0f; 
    float b = 2 * dot(D, O - C);
    float c = dot(O - C, O - C) - closest.r * closest.r;

    float delta = b * b - 4 * c;
    delta = sqrt(delta);
    float i1 = ((-b - delta) / 2.0f);
    float i2 = ((-b + delta) / 2.0f);
    float i = i1 < i2 ? i1 : i2;

    float3 P = O + i * D;
    float3 N = normalize(P - C);
    float3 V = normalize(O - P);


    /* Phong light model */
    float3 color = lights.ia * closest.ka * closest.color;
    for (int i = 0; i < lights.count; ++i) 
    {
        __syncthreads();

        float3 L = normalize(make_float3(lights.x[i], lights.y[i], lights.z[i]) - C);
        float3 R = normalize(2 * (dot(L, N)) * N - L);
        float3 lightColor = make_float3(lights.colorX[i], lights.colorY[i], lights.colorZ[i]);

        color = color +
            lights.id[i] * closest.kd * closest.color * lightColor * fmaxf(0.0f, dot(L, N)) +
            lights.is[i] * closest.ks * closest.color * lightColor * __powf(fmaxf(0.0f, dot(R, V)), closest.alpha);
    }

    /* clamp value */
    color = min(color, 1.0f);

    /* write data to texture */
    uchar4 writeData;
    writeData.x = (unsigned char)(color.x * 255.0f);
    writeData.y = (unsigned char)(color.y * 255.0f);
    writeData.z = (unsigned char)(color.z * 255.0f);
    writeData.w = 255;
    surf2Dwrite(writeData, surfaceObject, 4 * x, y);
}


#endif


