

#ifndef U1180779_CUDA_INLINE_H
#define U1180779_CUDA_INLINE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float3 min(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 min(const float3& a, const float f) {
    return make_float3(fminf(a.x, f), fminf(a.y, f), fminf(a.z, f));
}

__device__ __forceinline__ float3 max(const float3& a, const float f) {
    return make_float3(fmaxf(a.x, f), fmaxf(a.y, f), fmaxf(a.z, f));
}

__device__ __forceinline__ float3 operator*(const float3& a, const float f) {
    return make_float3(a.x * f, a.y * f, a.z * f);
}


__device__ __forceinline__ float3 fminf(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 fmaxf(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

#endif

