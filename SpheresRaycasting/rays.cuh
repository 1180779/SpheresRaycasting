
#ifndef U1180779_RAYS_CUH
#define U1180779_RAYS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ bool rayHit(float4 aabbMin, float4 aabbMax, float3 o)
{
    return aabbMin.x < o.x && o.x < aabbMax.x &&
        aabbMin.y < o.y && o.y < aabbMax.y &&
        o.z < aabbMax.z;
}

#endif