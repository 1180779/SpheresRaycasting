
#ifndef U1180779_UNIFIED_OBJECT_H
#define U1180779_UNIFIED_OBJECT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include "general.hpp"

/* underlying object types */
enum types {
    sphere,
    lightSource,
};

struct unifiedObject
{
    float x;
    float y;
    float z;
    float w;
    float r;

    types type;
    glm::ivec3 color;

    float ks;
    float kd;
    float alpha;
};


#endif

