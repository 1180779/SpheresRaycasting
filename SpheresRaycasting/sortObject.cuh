
#ifndef U1180779_SORT_OBJECT_H
#define U1180779_SORT_OBJECT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "unifiedObjects.hpp"
#include "general.hpp"

struct sortObject
{
    int* keys;
    int* index;
    unifiedObjects data;
    unifiedObjects temp;

    void sort();
    sortObject(unifiedObjects& data);
    ~sortObject();
};

sortObject::sortObject(unifiedObjects& data)
{
    this->data = data;
    xcudaMalloc(&keys, sizeof(int) * data.count);
    xcudaMalloc(&index, sizeof(int) * data.count);
    temp.dMalloc(data.count);
}

sortObject::~sortObject()
{
    xcudaFree(keys);
    xcudaFree(index);
    temp.dFree();
}

void sortObject::sort()
{
    thrust::sequence(index, index + data.count);
    thrust::sort(
        thrust::device_ptr<int>(keys),
        thrust::device_ptr<int>(keys + data.count),
        thrust::device_ptr<int>(index));

    dim3 blocks = dim3(data.count / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);

    copyToTempKernel<<<blocks, threads>>>(*this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    copyFromTempKernel<<<blocks, threads>>>(*this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
}

__global__ void copyToTempKernel(sortObject data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data.data.count)
        return;
    int index = data.keys[i];

    data.temp.x[i] = data.data.x[index];
    data.temp.y[i] = data.data.y[index];
    data.temp.z[i] = data.data.z[index];
    data.temp.w[i] = data.data.w[index];
    data.temp.r[i] = data.data.r[index];

    data.temp.type[i] = data.data.type[index];
    data.temp.color[i] = data.data.color[index];

    data.temp.ks[i] = data.data.ks[index];
    data.temp.kd[i] = data.data.kd[index];
    data.temp.alpha[i] = data.data.alpha[index];
}

__global__ void copyFromTempKernel(sortObject data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data.data.count)
        return;

    data.data.x[i] = data.temp.x[i];
    data.data.y[i] = data.temp.y[i];
    data.data.z[i] = data.temp.z[i];
    data.data.w[i] = data.temp.w[i];
    data.data.r[i] = data.temp.r[i];

    data.data.type[i] = data.temp.type[i];
    data.data.color[i] = data.temp.color[i];

    data.data.ks[i] = data.temp.ks[i];
    data.data.kd[i] = data.temp.kd[i];
    data.data.alpha[i] = data.temp.alpha[i];
}

#endif

