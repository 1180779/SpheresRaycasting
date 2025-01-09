
#ifndef U1180779_LIGHT_DATA_H
#define U1180779_LIGHT_DATA_H

#include <glm/glm.hpp>

struct lightData
{
    unsigned int count = 0;

    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* r = nullptr;
    glm::ivec3* color;


    void hMalloc(unsigned int c);
    void hFree();

    void dMalloc(unsigned int c);
    void dFree();

    void copyHostToDevice(const lightData& source);
};

#endif