
#include "dataObject.hpp"
#include "general.hpp"
#include <iostream>

void dataObject::generate(
    unsigned int count, 
    float rMin, float rMax, 
    float xMin, float xMax, 
    float yMin, float yMax, 
    float zMin, float zMax)
{
    m_objs.reserve(count);
    srand(static_cast <unsigned> (time(0)));
    for (int i = 0; i < count; ++i) {
        // TODO: check that spheres do not overlap
        unifiedObject obj;
        obj.x = xMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (xMax - xMin)));
        obj.y = yMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (yMax - yMin)));
        obj.z = zMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (zMax - zMin)));
        obj.w = 1.0f;
        obj.r = rMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (rMax - rMin)));
        
        obj.type = types::sphere;

        obj.color.x = rand() % 256;
        obj.color.y = rand() % 256;
        obj.color.z = rand() % 256;

        m_objs.push_back(obj);
        //std::cout << "x = " << obj.x << ", y = " << obj.y << ", z = " << obj.z << ", r = " << obj.r << std::endl;
    }
}

void dataObject::generateLights(
    unsigned int count,
    float rMin, float rMax,
    float xMin, float xMax,
    float yMin, float yMax,
    float zMin, float zMax)
{
    m_lights.reserve(count);
    mh_lights.hMalloc(count);
    for (int i = 0; i < count; ++i) {
        // TODO: check that spheres do not overlap
        unifiedObject obj;
        obj.x = xMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (xMax - xMin)));
        obj.y = yMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (yMax - yMin)));
        obj.z = zMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (zMax - zMin)));
        obj.w = 1.0f;
        obj.r = rMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (rMax - rMin)));

        obj.type = types::lightSource;

        obj.color.x = rand() % 256;
        obj.color.y = rand() % 256;
        obj.color.z = rand() % 256;

        m_lights.push_back(obj.light());
        mh_lights.x[i] = obj.x;
        mh_lights.y[i] = obj.y;
        mh_lights.z[i] = obj.z;
        mh_lights.w[i] = obj.w;
        mh_lights.cX[i] = obj.color.x;
        mh_lights.cY[i] = obj.color.y;
        mh_lights.cZ[i] = obj.color.z;

        m_objs.push_back(obj);
        //std::cout << "x = " << obj.x << ", y = " << obj.y << ", z = " << obj.z << ", r = " << obj.r << std::endl;
    }
    md_lights.dMalloc(count);
    md_lights.copyToDeviceFromHost(mh_lights);
}


