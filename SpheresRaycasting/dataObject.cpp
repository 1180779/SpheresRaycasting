
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

        obj.color.x = static_cast<float>(rand() % 256) / 255.0f;
        obj.color.y = static_cast<float>(rand() % 256) / 255.0f;
        obj.color.z = static_cast<float>(rand() % 256) / 255.0f;

        obj.ka = 0.3f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.5f - 0.3f)));
        obj.ks = 0.7f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1.0f - 0.7f)));;
        obj.kd = 0.5f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.7f - 0.5f)));
        obj.alpha = 2.0f;

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

        // not used yet
        obj.color.x = 255.0f / 2.0f;
        obj.color.y = 255.0f / 2.0f;
        obj.color.z = 255.0f / 2.0f;

        // save generated data
        mh_lights.x[i] = obj.x;
        mh_lights.y[i] = obj.y;
        mh_lights.z[i] = obj.z;
        mh_lights.w[i] = obj.w;

        mh_lights.is[i] = 0.2f;
        mh_lights.id[i] = 0.2f;

        m_objs.push_back(obj);
        //std::cout << "x = " << obj.x << ", y = " << obj.y << ", z = " << obj.z << ", r = " << obj.r << std::endl;
    }
    md_lights.dMalloc(count);
    md_lights.copyToDeviceFromHost(mh_lights);
}

void dataObject::freeLights()
{
    md_lights.dFree();
    mh_lights.hFree();
}


