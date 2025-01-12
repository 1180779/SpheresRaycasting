
#ifndef U1180779_DATA_OBJECT_H
#define U1180779_DATA_OBJECT_H

#include "unifiedObjects.cuh"
#include <cuda_runtime.h>

/* holds both device and host copies of the scene objects data */
class dataObject 
{
public:

    inline unsigned int hCount() { return mh_unified.count; }
    inline unsigned int dCount() { return md_unified.count; }
    
    /* generate spheres data */
    void generate(unsigned int count, float rMin, float rMax, float xMin, float xMax, float yMin, float yMax, float zMax, float zMin);
    
    void free();


    // mh - member host, md - memer device
    unifiedObjects mh_unified;
    unifiedObjects md_unified;
};

#endif