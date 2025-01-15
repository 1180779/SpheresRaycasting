
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

buffer::buffer(unsigned int maxWidth, unsigned int maxHeight)
    : m_maxWidth(maxWidth), m_maxHeight(maxHeight), m_shader(vertexTextureShader, fragmentTextureShader)
{
    /* for drawing texture on screen */
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    /* texture init */
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_maxWidth, m_maxHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);


    // register with cuda
    xcudaGraphicsGLRegisterImage(&m_cudaResource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);


    // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_maxWidth, m_maxHeight, GL_RGB, GL_FLOAT, pixelColors);
}

buffer::~buffer()
{
    glDeleteVertexArrays(1, &m_VAO);
    glDeleteBuffers(1, &m_VBO);

    glDeleteTextures(1, &m_texture);
}

void buffer::use()
{
    m_shader.use();
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void buffer::mapCudaResource()
{
    xcudaGraphicsMapResources(1, &m_cudaResource);
    cudaArray_t textureArray;
    xcudaGraphicsSubResourceGetMappedArray(&textureArray, m_cudaResource, 0, 0);

    // Set up a resource descriptor
    cudaResourceDesc resDesc = { };
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    // Create a surface object
    xcudaCreateSurfaceObject(&m_surfaceObject, &resDesc);
}

void buffer::unmapCudaResource()
{
    xcudaDestroySurfaceObject(m_surfaceObject);
    xcudaGraphicsUnmapResources(1, &m_cudaResource, 0);
}
