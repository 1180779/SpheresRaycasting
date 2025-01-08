
#include "spheresData.hpp"
#include "general.hpp"
#include <stdexcept>

void spheresData::hMalloc(unsigned int c)
{
    count = c;
    x = new float[count];
    y = new float[count];
    z = new float[count];
    r = new float[count];
}

void spheresData::hFree()
{
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] r;
}

void spheresData::dMalloc(unsigned int c)
{
    count = c;
    xcudaMalloc(&x, sizeof(float) * count);
    xcudaMalloc(&y, sizeof(float) * count);
    xcudaMalloc(&z, sizeof(float) * count);
    xcudaMalloc(&r, sizeof(float) * count);
}

void spheresData::dFree()
{
    xcudaFree(x);
    xcudaFree(y);
    xcudaFree(z);
    xcudaFree(r);
}

void spheresData::copyHostToDevice(const spheresData& source)
{
    if (source.count > count)
        throw std::invalid_argument("not enough memory");
    xcudaMemcpy(x, source.x, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(y, source.y, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(z, source.z, source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(r, source.r, source.count, cudaMemcpyHostToDevice);
}
