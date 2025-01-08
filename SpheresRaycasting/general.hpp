
#ifndef U1180779_SPHERE_H
#define U1180779_SPHERE_H

#include <cuda_runtime.h>
#include <iostream>

#define xcudaMalloc(devPtr, size) \
                        do { \
                            cudaError_t status = cudaMalloc(devPtr, size); \
                            if (status != cudaSuccess) { \
                                std::cerr << "error: " << cudaGetErrorString(status) << std::endl; \
                                std::cerr << "file: " << __FILE__ << "line: " << __LINE__ << std::endl; \
                                abort(); \
                            } \
                        } while(0)

#define xcudaFree(devPtr) \
                        do { \
                            cudaError_t status = cudaFree(devPtr); \
                            if (status != cudaSuccess) { \
                                std::cerr << "error: " << cudaGetErrorString(status) << std::endl; \
                                std::cerr << "file: " << __FILE__ << "line: " << __LINE__ << std::endl; \
                                abort(); \
                            } \
                        } while(0)

#define xcudaMemcpy(dst, src, count, kind) \
                        do { \
                            cudaError_t status = cudaMemcpy(dst, src, count, kind); \
                            if (status != cudaSuccess) { \
                                std::cerr << "error: " << cudaGetErrorString(status) << std::endl; \
                                std::cerr << "file: " << __FILE__ << "line: " << __LINE__ << std::endl; \
                                abort(); \
                            } \
                        } while(0)



#endif
