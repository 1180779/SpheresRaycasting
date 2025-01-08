
#ifndef U1180779_SPHERES_H
#define U1180779_SPHERES_H

#include "spheresData.hpp"
#include <cuda_runtime.h>

class spheres 
{
public:

    unsigned int hCount() { return mh_spheres.count; }
    unsigned int dCount() { return md_spheres.count; }
    void generate(unsigned int count, float rMin, float rMax, float xMin, float xMax, float yMin, float yMax, float zMax, float zMin);


private:
    // mh - member host, md - memer device
    spheresData mh_spheres;
    spheresData md_spheres;
};

#endif