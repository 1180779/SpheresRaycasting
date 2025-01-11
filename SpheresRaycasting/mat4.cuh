
#ifndef U1180779_MAT4_H
#define U1180779_MAT4_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct vec4 
{
    float data[4];

    vec4() { }

    vec4(float v0, float v1, float v2, float v3) {
        data[0] = v0;
        data[1] = v1;
        data[2] = v2;
        data[3] = v3;
    }

    float operator()(int i) const { return data[i]; }
    float& operator()(int i) { return data[i]; }
};

struct mat4 
{
    float data[16];

    __device__ __host__ float operator()(int row, int col) const {
        return data[row * 4 + col];
    }
    
    __device__ __host__ float& operator()(int row, int col) {
        return data[row * 4 + col];
    }

    __device__ __host__ mat4 operator*(const mat4& other) const {
        mat4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result(i, j) = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    __device__ __host__ vec4 operator*(const vec4& v) const {
        vec4 result;
        for (int i = 0; i < 4; ++i) {
            result(i) = 0.0f;
            for (int j = 0; j < 4; ++j) {
                result(j) += (*this)(i, j) * v(j);
            }
        }
        return result;
    }

    __host__ mat4 operator=(const glm::mat4& mat) {
        for (int i = 0; i < 4; ++i) {
            for(int j = 0; j < 4; ++j) {
                (*this)(i, j) = mat[i][j];
            }
        }
        
    }
};

#endif