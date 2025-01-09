
#include "lightData.hpp"
#include "general.hpp"
#include <stdexcept>

void lightData::hMalloc(unsigned int c)
{
    count = c;
    x = new float[count];
    y = new float[count];
    z = new float[count];
    r = new float[count];
    color = new glm::ivec3[count];
}

void lightData::hFree()
{
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] r;
    delete[] color;
}

void lightData::dMalloc(unsigned int c)
{
    count = c;
    xcudaMalloc(&x, sizeof(float) * count);
    xcudaMalloc(&y, sizeof(float) * count);
    xcudaMalloc(&z, sizeof(float) * count);
    xcudaMalloc(&r, sizeof(float) * count);
    xcudaMalloc(&color, sizeof(glm::ivec3) * count);
}

void lightData::dFree()
{
    xcudaFree(x);
    xcudaFree(y);
    xcudaFree(z);
    xcudaFree(r);
    xcudaFree(color);
}

void lightData::copyHostToDevice(const lightData& source)
{
    if (source.count > count)
        throw std::invalid_argument("not enough memory");
    xcudaMemcpy(x, source.x, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(y, source.y, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(z, source.z, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(r, source.r, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(color, source.color, source.count, cudaMemcpyHostToDevice);
}
