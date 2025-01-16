
#ifndef U1180779_UNIFIED_OBJECTS_H
#define U1180779_UNIFIED_OBJECTS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "unifiedObject.hpp"
#include "cudaWrappers.hpp"

struct unifiedObjects
{
    unsigned int count;

    float *x;
    float *y;
    float *z;
    float *r;

    types* type;
    float* colorX;
    float* colorY;
    float* colorZ;

    float* ka;
    float* ks;
    float* kd;
    float* alpha;

    __device__ __forceinline__ void set(const unifiedObject& other, int i);

    void dMalloc();
    void dFree();
};

__device__ __forceinline__ void unifiedObjects::set(const unifiedObject& other, int i)
{
    x[i] = other.x;
    y[i] = other.y;
    z[i] = other.z;
    r[i] = other.r;

    type[i] = other.type;
    colorX[i] = other.color.x;
    colorY[i] = other.color.y;
    colorZ[i] = other.color.z;

    ka[i] = other.ka;
    ks[i] = other.ks;
    kd[i] = other.kd;
    alpha[i] = other.alpha;
}

void unifiedObjects::dMalloc()
{
    xcudaMalloc(&x, sizeof(float) * count);
    xcudaMalloc(&y, sizeof(float) * count);
    xcudaMalloc(&z, sizeof(float) * count);
    xcudaMalloc(&r, sizeof(float) * count);
    
    xcudaMalloc(&type, sizeof(types) * count);
    
    xcudaMalloc(&colorX, sizeof(float) * count);
    xcudaMalloc(&colorY, sizeof(float) * count);
    xcudaMalloc(&colorZ, sizeof(float) * count);
    
    xcudaMalloc(&ka, sizeof(float) * count);
    xcudaMalloc(&ks, sizeof(float) * count);
    xcudaMalloc(&kd, sizeof(float) * count);
    xcudaMalloc(&alpha, sizeof(float) * count);
}

void unifiedObjects::dFree()
{
    xcudaFree(x);
    xcudaFree(y);
    xcudaFree(z);
    xcudaFree(r);

    xcudaFree(type);
    
    xcudaFree(colorX);
    xcudaFree(colorY);
    xcudaFree(colorZ);
    
    xcudaFree(ka);
    xcudaFree(ks);
    xcudaFree(kd);
    xcudaFree(alpha);
}

#endif
