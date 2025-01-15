
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


    float3 color;
};

struct unifiedObject
{
    float x;
    float y;
    float z;
    float w;
    float r;

    types type;
    float3 color;

    float ka;
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
        return l;
    }
};

struct dLights
{
    unsigned int count;

    float* x;
    float* y;
    float* z;
    float* w;

    float* is; // specular light intensity ([0, 1])
    float* id; // diffuse light intensity ([0, 1])

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
        
        is = new float[count];
        id = new float[count];
    }

    void hFree() 
    {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] w;

        delete[] is;
        delete[] id;
    }

    void dMalloc(unsigned int count) 
    {
        this->count = count;
        xcudaMalloc(&x, sizeof(float) * count);
        xcudaMalloc(&y, sizeof(float) * count);
        xcudaMalloc(&z, sizeof(float) * count);
        xcudaMalloc(&w, sizeof(float) * count);
        
        xcudaMalloc(&is, sizeof(float) * count);
        xcudaMalloc(&id, sizeof(float) * count);
    }
    
    void dFree() 
    {
        xcudaFree(x);
        xcudaFree(y);
        xcudaFree(z);
        xcudaFree(w);

        xcudaFree(is);
        xcudaFree(id);
    }

    void copyToDeviceFromHost(const dLights& other) 
    {
        count = other.count;
        xcudaMemcpy(x, other.x, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(y, other.y, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(z, other.z, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(w, other.w, sizeof(float) * count, cudaMemcpyHostToDevice);
        
        xcudaMemcpy(is, other.is, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(id, other.id, sizeof(float) * count, cudaMemcpyHostToDevice);
    }
};

#endif

