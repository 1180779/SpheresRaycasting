
#include "spheresData.hpp"
#include "general.hpp"
#include <stdexcept>

void spheresData::hMalloc(unsigned int c)
{
    count = c;
    x = new float[count];
    y = new float[count];
    z = new float[count];
    w = new float[count];
    r = new float[count];
    color = new glm::ivec3[count];
}

void spheresData::hFree()
{
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;
    delete[] r;
    delete[] color;
}

void spheresData::dMalloc(unsigned int c)
{
    count = c;
    xcudaMalloc(&x, sizeof(float) * count);
    xcudaMalloc(&y, sizeof(float) * count);
    xcudaMalloc(&z, sizeof(float) * count);
    xcudaMalloc(&w, sizeof(float) * count);
    xcudaMalloc(&r, sizeof(float) * count);
    xcudaMalloc(&color, sizeof(glm::ivec3) * count);
}

void spheresData::dFree()
{
    xcudaFree(x);
    xcudaFree(y);
    xcudaFree(z);
    xcudaFree(w);
    xcudaFree(r);
    xcudaFree(color);
}

void spheresData::copyHostToDevice(const spheresData& source)
{
    if (source.count > count)
        throw std::invalid_argument("not enough memory");
    xcudaMemcpy(x, source.x, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(y, source.y, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(z, source.z, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(w, source.w, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(r, source.r, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(color, source.color, sizeof(glm::ivec3) * source.count, cudaMemcpyHostToDevice);
}

void spheresData::copyDeviceToHost(const spheresData& source)
{
    if (source.count > count)
        throw std::invalid_argument("not enough memory");
    xcudaMemcpy(x, source.x, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(y, source.y, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(z, source.z, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(w, source.w, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(r, source.r, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(color, source.color, sizeof(glm::ivec3) * source.count, cudaMemcpyDeviceToHost);
}
