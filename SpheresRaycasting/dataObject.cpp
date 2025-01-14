
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


