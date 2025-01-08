
#ifndef U1180779_BUFFER_H
#define U1180779_BUFFER_H

#define NULL 0
#include "glad/glad.h"
#include <cuda.h>
#include <cuda_gl_interop.h>

class buffer 
{
    buffer(unsigned int maxWidth = 1920, unsigned int maxHeight = 1080);
    ~buffer();
    void use();
    void mapCudaResource();
    void unmapCudaResource();

    unsigned int m_maxWidth;
    unsigned int m_maxHeight;
    GLuint m_texture;
    cudaGraphicsResource* m_cudaResource;
    cudaSurfaceObject_t m_surfaceObject;
};

#endif

buffer::buffer(unsigned int maxWidth, unsigned int maxHeight)
    : m_maxWidth(maxWidth), m_maxHeight(maxHeight)
{
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_maxWidth, m_maxHeight, 0, GL_RGB, GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // register with cuda
    cudaGraphicsGLRegisterImage(&m_cudaResource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);


    // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_maxWidth, m_maxHeight, GL_RGB, GL_FLOAT, pixelColors);
}

buffer::~buffer()
{
    glDeleteTextures(1, &m_texture);
}

void buffer::use()
{
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void buffer::mapCudaResource()
{
    cudaGraphicsMapResources(1, &m_cudaResource, 0);
    cudaArray_t textureArray;
    cudaGraphicsSubResourceGetMappedArray(&textureArray, m_cudaResource, 0, 0);
    //cudaMemcpyToArray(textureArray, 0, 0, pixelColors, sizeInBytes, cudaMemcpyDeviceToDevice);

    // Set up a resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    // Create a surface object
    cudaCreateSurfaceObject(&m_surfaceObject, &resDesc);
}

void buffer::unmapCudaResource()
{
    cudaDestroySurfaceObject(m_surfaceObject);
    cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
}
