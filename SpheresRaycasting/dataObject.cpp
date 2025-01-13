
#include "dataObject.hpp"
#include "general.hpp"
#include <iostream>

void dataObject::generate(unsigned int count, float rMin, float rMax, float xMin, float xMax, float yMin, float yMax, float zMax, float zMin)
{
    mh_unified.hMalloc(count);
    srand(static_cast <unsigned> (time(0)));
    for (int i = 0; i < count; ++i) {
        // TODO: check that spheres do not overlap
        mh_unified.x[i] = xMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (xMax - xMin)));
        mh_unified.y[i] = yMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (yMax - yMin)));
        mh_unified.z[i] = zMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (zMax - zMin)));
        mh_unified.w[i] = 1.0f;
        mh_unified.r[i] = rMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (rMax - rMin)));
        
        mh_unified.type[i] = types::sphere;

        mh_unified.color[i].x = rand() % 256;
        mh_unified.color[i].y = rand() % 256;
        mh_unified.color[i].z = rand() % 256;

        //std::cout << "x = " << mh_unified.x[i] << ", y = " << mh_unified.y[i] << ", z = " << mh_unified.z[i] << ", r = " << mh_unified.r[i] << std::endl;
    }

    md_unified.dMalloc(count);
    md_unified.copyHostToDevice(mh_unified);
}

void dataObject::free()
{
    mh_unified.hFree();
    md_unified.dFree();
}

