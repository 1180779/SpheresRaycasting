
#ifndef U1180779_UNIFIED_OBJECTS_H
#define U1180779_UNIFIED_OBJECTS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include "cudaWrappers.hpp"

/* underlying object types */
enum types {
    sphere,
    lightSource,
};

struct light
{
    float x;
    float y;
    float z;
    float w;

    float3 color;
};

struct unifiedObject
{
    float x;
    float y;
    float z;
    float w;
    float r;

    types type;
    float3 color;

    float ka;
    float ks;
    float kd;
    float alpha;

    light light() {
        ::light l;
        l.x = x;
        l.y = y;
        l.z = z;
        l.w = w;
        l.color = color;
        return l;
    }
};

#endif

