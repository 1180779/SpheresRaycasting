
#include "unifiedObjects.cuh"
#include <stdexcept>

void unifiedObjects::hMalloc(unsigned int c)
{
    count = c;
    x = new float[count];
    y = new float[count];
    z = new float[count];
    w = new float[count];
    r = new float[count];

    type = new types[count];
    color = new glm::ivec3[count];

    ks = new float[count];
    kd = new float[count];
    alpha = new float[count];
}

void unifiedObjects::hFree()
{
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;
    delete[] r;

    delete[] type;
    delete[] color;

    delete[] ks;
    delete[] kd;
    delete[] alpha;
}

void unifiedObjects::dMalloc(unsigned int c)
{
    count = c;
    xcudaMalloc(&x, sizeof(float) * count);
    xcudaMalloc(&y, sizeof(float) * count);
    xcudaMalloc(&z, sizeof(float) * count);
    xcudaMalloc(&w, sizeof(float) * count);
    xcudaMalloc(&r, sizeof(float) * count);

    xcudaMalloc(&color, sizeof(glm::ivec3) * count);
    xcudaMalloc(&type, sizeof(types) * count);

    xcudaMalloc(&kd, sizeof(float) * count);
    xcudaMalloc(&ks, sizeof(float) * count);
    xcudaMalloc(&alpha, sizeof(float) * count);
}

void unifiedObjects::dFree()
{
    xcudaFree(x);
    xcudaFree(y);
    xcudaFree(z);
    xcudaFree(w);
    xcudaFree(r);

    xcudaFree(type);
    xcudaFree(color);
    
    xcudaFree(ks);
    xcudaFree(kd);
    xcudaFree(alpha);
}

void unifiedObjects::copyHostToDevice(const unifiedObjects& source)
{
    if (source.count > count)
        throw std::invalid_argument("not enough memory");
    xcudaMemcpy(x, source.x, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(y, source.y, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(z, source.z, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(w, source.w, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(r, source.r, sizeof(float) * source.count, cudaMemcpyHostToDevice);


    xcudaMemcpy(type, source.type, sizeof(types) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(color, source.color, sizeof(glm::ivec3) * source.count, cudaMemcpyHostToDevice);

    xcudaMemcpy(ks, source.ks, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(kd, source.kd, sizeof(float) * source.count, cudaMemcpyHostToDevice);
    xcudaMemcpy(alpha, source.alpha, sizeof(float) * source.count, cudaMemcpyHostToDevice);
}

void unifiedObjects::copyDeviceToHost(const unifiedObjects& source)
{
    if (source.count > count)
        throw std::invalid_argument("not enough memory");
    xcudaMemcpy(x, source.x, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(y, source.y, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(z, source.z, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(w, source.w, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(r, source.r, sizeof(float) * source.count, cudaMemcpyDeviceToHost);

    xcudaMemcpy(type, source.type, sizeof(types) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(color, source.color, sizeof(glm::ivec3) * source.count, cudaMemcpyDeviceToHost);

    xcudaMemcpy(ks, source.ks, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(kd, source.kd, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
    xcudaMemcpy(alpha, source.alpha, sizeof(float) * source.count, cudaMemcpyDeviceToHost);
}
