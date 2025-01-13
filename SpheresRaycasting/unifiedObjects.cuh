
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

struct unifiedObjectsSingle
{
    __host__ __device__ unifiedObjectsSingle(float& x, float& y, float& z, float& w, float& r, 
        types& type, glm::ivec3& color, float& ks, float& kd, float& alpha)
        : x(x), y(y), z(z), w(w), r(r), type(type), color(color), ks(ks), kd(kd), alpha(alpha) { }

    float& x;
    float& y;
    float& z;
    float& w;
    float& r;

    types& type;
    glm::ivec3& color;

    float& ks;
    float& kd;
    float& alpha;
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

    __host__ __device__ unifiedObject operator=(const unifiedObjectsSingle& other) 
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        r = other.r;

        type = other.type;
        color = other.color;

        ks = other.ks;
        kd = other.kd;
        alpha = other.alpha;
    }
};


/* constains data of all objects
    in the scene in one unified structure 
*/
struct unifiedObjects
{
    unsigned int count = 0; /* object count */

    /* shared values (both sphere and light source (also sphere)) */
    float* x = nullptr; /* x */
    float* y = nullptr; /* y */
    float* z = nullptr; /* z */
    float* w = nullptr; /* w */
    float* r = nullptr; /* sphere radius */

    types* type = nullptr; /* type, equal to one of enum values */
    glm::ivec3* color = nullptr; /* sphere or light color */

    /* sphere attributes */
    float* ks = nullptr; /* ks */
    float* kd = nullptr; /* kd */
    float* alpha = nullptr; /* alpha */

    float3 aabbMin;
    float3 aabbMax;
    float3 aabbRange;

    /* malloc host memory */
    void hMalloc(unsigned int c);
    
    /* free host memory */
    void hFree();

    /* malloc device memory */
    void dMalloc(unsigned int c);

    /* free device memory */
    void dFree();

    /* copy from source to current object; from host to device */
    void copyHostToDevice(const unifiedObjects& source);

    /* copy from source to current object; from device to hosts */
    void copyDeviceToHost(const unifiedObjects& source);

    void findAABB();

    __host__ __device__ unifiedObjectsSingle operator()(int i) 
    {
        return unifiedObjectsSingle(x[i], y[i], z[i], w[i], r[i], type[i], color[i], ks[i], kd[i], alpha[i]);
    }
};

#endif

