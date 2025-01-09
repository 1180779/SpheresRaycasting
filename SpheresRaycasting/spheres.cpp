
#include "spheres.hpp"
#include "general.hpp"
#include <iostream>

void spheres::generate(unsigned int count, float rMin, float rMax, float xMin, float xMax, float yMin, float yMax, float zMax, float zMin)
{
    mh_spheres.hMalloc(count);
    srand(static_cast <unsigned> (time(0)));
    for (int i = 0; i < count; ++i) {
        // TODO: check that spheres do not overlap
        mh_spheres.x[i] = xMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (xMax - xMin)));
        mh_spheres.y[i] = yMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (yMax - yMin)));
        mh_spheres.z[i] = zMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (zMax - zMin)));
        mh_spheres.w[i] = 1.0f;
        mh_spheres.r[i] = rMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (rMax - rMin)));
        
        mh_spheres.color[i].x = rand() % 256;
        mh_spheres.color[i].y = rand() % 256;
        mh_spheres.color[i].z = rand() % 256;

        std::cout << "x = " << mh_spheres.x[i] << ", y = " << mh_spheres.y[i] << ", z = " << mh_spheres.z[i] << ", r = " << mh_spheres.r[i] << std::endl;
    }

    md_spheres.dMalloc(count);
    md_spheres.copyHostToDevice(mh_spheres);
}
