
#ifndef U1180779_DATA_OBJECT_H
#define U1180779_DATA_OBJECT_H

#include "unifiedObjects.cuh"
#include <cuda_runtime.h>
#include <vector>

/* holds both device and host copies of the scene objects data */
class dataObject 
{
public:

    inline unsigned int size() { return m_objs.size(); }
    inline void clear() { m_objs.clear(); }

    /* generate spheres data */
    void generate(
        unsigned int count, 
        float rMin, float rMax, 
        float xMin, float xMax, 
        float yMin, float yMax, 
        float zMin, float zMax);

    // mh - member host, md - memer device
    std::vector<unifiedObject> m_objs;
};

#endif