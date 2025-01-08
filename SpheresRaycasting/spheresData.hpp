
#ifndef U1180779_SPHERES_DATA_H
#define U1180779_SPHERES_DATA_H

struct spheresData
{
    unsigned int count = 0;
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* r = nullptr;

    void hMalloc(unsigned int c);
    void hFree();

    void dMalloc(unsigned int c);
    void dFree();

    void copyHostToDevice(const spheresData& source);
};

#endif