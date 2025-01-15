
#ifndef U1180779_LIGHTS_H
#define U1180779_LIGHTS_H

#include "cudaWrappers.hpp"

struct lights
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

    void copyToDeviceFromHost(const lights& other)
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