
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

    float* colorX;
    float* colorY;
    float* colorZ;

    float ia = 0.5f;
    float4 clearColor;

    void hMalloc(unsigned int count)
    {
        this->count = count;

        x = new float[count];
        y = new float[count];
        z = new float[count];
        w = new float[count];

        is = new float[count];
        id = new float[count];

        colorX = new float[count];
        colorY = new float[count];
        colorZ = new float[count];
    }

    void hFree()
    {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] w;

        delete[] is;
        delete[] id;

        delete[] colorX;
        delete[] colorY;
        delete[] colorZ;
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

        xcudaMalloc(&colorX, sizeof(float) * count);
        xcudaMalloc(&colorY, sizeof(float) * count);
        xcudaMalloc(&colorZ, sizeof(float) * count);
    }

    void dFree()
    {
        xcudaFree(x);
        xcudaFree(y);
        xcudaFree(z);
        xcudaFree(w);

        xcudaFree(is);
        xcudaFree(id);

        xcudaFree(colorX);
        xcudaFree(colorY);
        xcudaFree(colorZ);
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

        xcudaMemcpy(colorX, other.colorX, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(colorY, other.colorY, sizeof(float) * count, cudaMemcpyHostToDevice);
        xcudaMemcpy(colorZ, other.colorZ, sizeof(float) * count, cudaMemcpyHostToDevice);
    }
};

#endif