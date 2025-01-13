
#include <stdexcept>
#include "unifiedObjects.cuh"
#include "cudaInline.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include "timer.hpp"

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

//__device__ float fatomicMax(float* addr, float value)
//{
//    float old = *addr, assumed;
//    if (old >= value)
//        return old;
//    do
//    {
//        assumed = old;
//        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
//    } while (old != assumed);
//    return old;
//}
//
//__device__ float fatomicMin(float* addr, float value)
//{
//    float old = *addr, assumed;
//    if (old <= value)
//        return old;
//    do
//    {
//        assumed = old;
//        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
//    } while (old != assumed);
//    return old;
//}
//
//__global__ void minmaxKernel(const float* arr, int count, float* minOut, float* maxOut) {
//    extern __shared__ float sharedMinMax[];
//    float* s_min = sharedMinMax;
//    float* s_max = sharedMinMax + blockDim.x;
//
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    int localTid = threadIdx.x;
//
//    if (tid < count) {
//        s_min[localTid] = arr[tid];
//        s_max[localTid] = arr[tid];
//    }
//    else {
//        s_min[localTid] = FLT_MAX;
//        s_max[localTid] = -FLT_MAX;
//    }
//    __syncthreads();
//
//    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//        if (localTid < offset && tid + offset < count) {
//            s_min[localTid] = fminf(s_min[localTid], s_min[localTid + offset]);
//            s_max[localTid] = fmaxf(s_max[localTid], s_max[localTid + offset]);
//        }
//        __syncthreads();
//    }
//
//    if (localTid == 0) {
//        *minOut = s_min[0];
//        *maxOut = s_max[0];
//    }
//}
//
//
//


void unifiedObjects::findAABB()
{
    timer t;

    //float* cRes;
    //cudaMalloc(&cRes, sizeof(float) * 2);

    //// custom kernel for comparison
    //t.start();
    //int blockSize = 256;
    //int numBlocks = (count + blockSize - 1) / blockSize;
    //minmaxKernel << <numBlocks, blockSize, 2 * blockSize * sizeof(float) >> > (x, count, cRes, cRes + 1);
    //xcudaDeviceSynchronize();
    //xcudaGetLastError();
    //t.stop("\tcustom minmax kernel");

    t.start();
    auto xP = thrust::minmax_element(thrust::device_ptr<float>(x), thrust::device_ptr<float>(x + count));
    t.stop("\tminmax_element x");

    t.start();
    auto yP = thrust::minmax_element(thrust::device_ptr<float>(y), thrust::device_ptr<float>(y + count));
    t.stop("\tminmax_element y");
    
    t.start();
    auto zP = thrust::minmax_element(thrust::device_ptr<float>(z), thrust::device_ptr<float>(z + count));
    t.stop("\tminmax_element z");
    
    //float minCustom;
    //float maxCustom;
    //xcudaMemcpy(&minCustom, cRes, sizeof(float), cudaMemcpyDeviceToHost);
    //xcudaMemcpy(&maxCustom, cRes + 1, sizeof(float), cudaMemcpyDeviceToHost);

    //xcudaFree(cRes);

    t.start();
    xcudaMemcpy(&aabbMin.x, xP.first.get(), sizeof(float), cudaMemcpyDeviceToHost);
    xcudaMemcpy(&aabbMax.x, xP.second.get(), sizeof(float), cudaMemcpyDeviceToHost);
    xcudaMemcpy(&aabbMin.y, yP.first.get(), sizeof(float), cudaMemcpyDeviceToHost);
    xcudaMemcpy(&aabbMax.y, yP.second.get(), sizeof(float), cudaMemcpyDeviceToHost);
    xcudaMemcpy(&aabbMin.z, zP.first.get(), sizeof(float), cudaMemcpyDeviceToHost);
    xcudaMemcpy(&aabbMax.z, zP.second.get(), sizeof(float), cudaMemcpyDeviceToHost);
    t.stop("\tcopy back");
}



