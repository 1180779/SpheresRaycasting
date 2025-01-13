
#include "sortObject.cuh"

#include <cub/device/device_radix_sort.cuh>

void radix_sort(
    int num_items,
    unsigned int* d_keys_in, unsigned int* d_keys_out,
    unsigned int* d_values_in, unsigned int* d_values_out)
{
    // determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // allocate temporary storage
    xcudaMalloc(&d_temp_storage, temp_storage_bytes);
    // run sorting operation
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // free temporary storage
    xcudaFree(d_temp_storage);
}

void sortObject::malloc(unifiedObjects& data)
{
    this->data = data;
    xcudaMalloc(&keysIn, sizeof(unsigned int) * data.count);
    xcudaMalloc(&indexIn, sizeof(int) * data.count);
    xcudaMalloc(&keysOut, sizeof(unsigned int) * data.count);
    xcudaMalloc(&indexOut, sizeof(int) * data.count);
    temp.dMalloc(data.count);
}

void sortObject::free()
{
    xcudaFree(keysIn);
    xcudaFree(indexIn);
    xcudaFree(keysOut);
    xcudaFree(indexOut);
    temp.dFree();
}

void sortObject::sort()
{
    thrust::sequence(
        thrust::device_ptr<int>(indexIn), 
        thrust::device_ptr<int>(indexIn + data.count));

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        keysIn, keysOut, indexIn, indexOut, data.count);

    xcudaMalloc(&d_temp_storage, temp_storage_bytes);

    // run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        keysIn, keysOut, indexIn, indexOut, data.count);

    // free memory
    xcudaFree(d_temp_storage);

    dim3 blocks = dim3(data.count / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);

    copyToTempKernel << <blocks, threads >> > (*this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    copyFromTempKernel << <blocks, threads >> > (*this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
}

__global__ void copyToTempKernel(sortObject data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data.data.count)
        return;
    int index = data.indexOut[i];

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

