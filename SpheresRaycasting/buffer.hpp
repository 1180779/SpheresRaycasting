
#ifndef U1180779_BUFFER_H
#define U1180779_BUFFER_H

#define NULL 0
#include "glad/glad.h"
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "cudaWrappers.hpp"
#include "shader.hpp"
#include "shaderSource.hpp"

class buffer 
{
public:
    buffer(unsigned int maxWidth = 1920, unsigned int maxHeight = 1080);
    ~buffer();
    void use();
    void mapCudaResource();
    void unmapCudaResource();

    unsigned int m_maxWidth;
    unsigned int m_maxHeight;
    GLuint m_texture;
    cudaGraphicsResource_t m_cudaResource;
    cudaSurfaceObject_t m_surfaceObject;

    /* drawing the texture onto display */
    shader m_shader;
    float m_vertices[6 * 5] = {
        -1.0f, -1.0f, 0.0f,   0.0f,  0.0f,
        -1.0f,  1.0f, 0.0f,   0.0f,  1.0f,
         1.0f, -1.0f, 0.0f,   1.0f,  0.0f,

         1.0f, -1.0f, 0.0f,   1.0f,  0.0f,
         1.0f,  1.0f, 0.0f,   1.0f,  1.0f,
        -1.0f,  1.0f, 0.0f,   0.0f,  1.0f,
    };
    GLuint m_VBO;
    GLuint m_VAO;
};

#endif
