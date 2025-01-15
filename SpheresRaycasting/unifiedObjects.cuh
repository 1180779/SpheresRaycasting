
#ifndef U1180779_UNIFIED_OBJECT_H
#define U1180779_UNIFIED_OBJECT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include "general.hpp"

/* underlying object types */
enum types {
    sphere,
    lightSource,
};

struct light
{
    float x;
    float y;
    float z;
    float w;


    glm::ivec3 color;
};

struct unifiedObject
{
    float x;
    float y;
    float z;
    float w;
    float r;

    types type;
    glm::ivec3 color;

    float ks;
    float kd;
    float alpha;

    light light() {
        ::light l;
        l.x = x;
        l.y = y;
        l.z = z;
        l.w = w;
        l.color = color;
    }
};

struct dLights
{
    unsigned int count;

    float* x;
    float* y;
    float* z;
    float* w;

    int* cX; // color x
    int* cY; // color y
    int* cZ; // color z

    //light operator()(int i) 
    //{
    //    light l;
    //    l.x = x[i];
    //    l.y = y[i];
    //    l.z = z[i];
    //    l.w = w[i];
    //    l.color = glm::ivec3(l.color);
    //    return l;
    //}

    void hMalloc(unsigned int count) 
    {
        this->count = count;

        x = new float[count];
        y = new float[count];
        z = new float[count];
        w = new float[count];
        
        cX = new int[count];
        cY = new int[count];
        cZ = new int[count];
    }

    void hFree() 
    {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] w;

        delete[] cX;
        delete[] cY;
        delete[] cZ;
    }

    void dMalloc(unsigned int count) 
    {
        this->count = count;
        xcudaMalloc(&x, sizeof(float) * count);
        xcudaMalloc(&y, sizeof(float) * count);
        xcudaMalloc(&z, sizeof(float) * count);
        xcudaMalloc(&w, sizeof(float) * count);
        
        xcudaMalloc(&cX, sizeof(int) * count);
        xcudaMalloc(&cY, sizeof(int) * count);
        xcudaMalloc(&cZ, sizeof(int) * count);
    }
    
    void dFree() 
    {
        xcudaFree(x);
        xcudaFree(y);
        xcudaFree(z);
        xcudaFree(w);

        xcudaFree(cX);
        xcudaFree(cY);
        xcudaFree(cZ);
    }

    void copyToDeviceFromHost(const dLights& other) 
    {
        xcudaMemcpy(x, other.x, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(y, other.y, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(z, other.z, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(w, other.w, sizeof(float) * count, cudaMemcpyHostToDevice);
        
        xcudaMemcpy(cX, other.cX, sizeof(int) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(cY, other.cY, sizeof(int) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(cZ, other.cZ, sizeof(int) * count, cudaMemcpyHostToDevice);
    }
};

#endif

