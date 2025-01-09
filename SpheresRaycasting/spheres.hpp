
#ifndef U1180779_SPHERES_H
#define U1180779_SPHERES_H

#include "spheresData.hpp"
#include <cuda_runtime.h>

class spheres 
{
public:

    inline unsigned int hCount() { return mh_spheres.count; }
    inline unsigned int dCount() { return md_spheres.count; }
    
    void generate(unsigned int count, float rMin, float rMax, float xMin, float xMax, float yMin, float yMax, float zMax, float zMin);
    inline void free() 
    { 
        mh_spheres.hFree(); 
        md_spheres.dFree();
    }


    // mh - member host, md - memer device
    spheresData mh_spheres;
    spheresData md_spheres;
};

#endif