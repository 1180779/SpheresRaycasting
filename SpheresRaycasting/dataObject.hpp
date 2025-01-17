
#ifndef U1180779_DATA_OBJECT_H
#define U1180779_DATA_OBJECT_H

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

#include "randomValueGenerator.hpp"
#include "materialGenerator.hpp"
#include "unifiedObject.hpp"
#include "lights.hpp"

/* holds both device and host copies of the scene objects data */
class dataObject 
{
public:

    inline unsigned int size() { return m_objs.size(); }
    inline void clear() { m_objs.clear(); }

    /* generate spheres data */
    void generate(
        unsigned int count, 
        range rR, range xR,
        range yR, range zR,
        materialGenerator::type t = materialGenerator::type::reflective);

    void generateLights(
        unsigned int count,
        range rR, range xR,
        range yR, range zR,
        range isR, range idR
    );

    void freeLights();
    
    // mh - member host, md - memer device
    std::vector<unifiedObject> m_objs;
    lights mh_lights;
    lights md_lights;

};

#endif