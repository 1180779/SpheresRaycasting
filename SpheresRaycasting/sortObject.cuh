
#ifndef U1180779_SORT_OBJECT_H
#define U1180779_SORT_OBJECT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "unifiedObjects.cuh"
#include "general.hpp"

struct sortObject
{
    // need two copies for radix sort from cub
    unsigned int* keysIn;
    unsigned int* keysOut;
    int* indexIn;
    int* indexOut;

    unifiedObjects data;
    unifiedObjects temp;

    void sort();
    void malloc(unifiedObjects& data);
    void free();
};

__global__ void copyToTempKernel(sortObject data);
__global__ void copyFromTempKernel(sortObject data);

#endif

