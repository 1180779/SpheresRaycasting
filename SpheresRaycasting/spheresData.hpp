
#ifndef U1180779_SPHERES_DATA_H
#define U1180779_SPHERES_DATA_H

#include <glm/glm.hpp>

struct spheresData
{
    unsigned int count = 0;
    
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* w = nullptr;
    float* r = nullptr;

    float* ks = nullptr;
    float* kd = nullptr;
    float* alpha = nullptr;
    glm::ivec3* color = nullptr;



    void hMalloc(unsigned int c);
    void hFree();

    void dMalloc(unsigned int c);
    void dFree();

    void copyHostToDevice(const spheresData& source);
};

struct sphereData {
    float x;
    float y;
    float z;
    float r;
    glm::vec3 color;
};

#endif