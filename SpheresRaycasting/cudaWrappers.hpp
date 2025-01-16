
#ifndef U1180779_CUDA_WRAPPERS_H
#define U1180779_CUDA_WRAPPERS_H

#include <cuda_runtime.h>
#include <stdio.h>

/* set of cuda wrapper macros with the motto succeed or die
 names are the same as cuda functions, but being with 'x' */


#define BLOCK_SIZE1D 256
#define BLOCK_SIZE2D 16

#define xcuda(operation) do { \
            cudaError_t status = operation; \
            if(status != cudaSuccess) \
            { \
                fprintf(stderr, "error: %s\n",  cudaGetErrorString(status)); \
                fprintf(stderr, "file: %s, line: %d\n", __FILE__, __LINE__); \
                abort(); \
            } \
        } while(0)

#define xcudaSetDevice(device) \
    xcuda(cudaSetDevice(device))

#define xcudaDeviceSynchronize() \
    xcuda(cudaDeviceSynchronize())

#define xcudaGetLastError() \
    xcuda(cudaGetLastError())

#define xcudaMalloc(devPtr, size) \
    xcuda(cudaMalloc(devPtr, size))

#define xcudaFree(devPtr) \
    xcuda(cudaFree(devPtr))

#define xcudaMemcpy(dst, src, count, kind) \
    xcuda(cudaMemcpy(dst, src, count, kind))

#define xcudaGraphicsGLRegisterImage(resource, image, target, flags) \
    xcuda(cudaGraphicsGLRegisterImage(resource, image, target, flags))

#define xcudaGraphicsMapResources(count, resources) \
    xcuda(cudaGraphicsMapResources(count, resources))

#define xcudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel) \
    xcuda(cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel))

#define xcudaCreateSurfaceObject(pSurfObject, pResDesc) \
    xcuda(cudaCreateSurfaceObject(pSurfObject, pResDesc))

#define xcudaDestroySurfaceObject(surfObject) \
    xcuda(cudaDestroySurfaceObject(surfObject))

#define xcudaGraphicsUnmapResources(count, resources) \
    xcuda(cudaGraphicsUnmapResources(count, resources))




#endif
