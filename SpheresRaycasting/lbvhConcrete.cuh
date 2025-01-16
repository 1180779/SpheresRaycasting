
#ifndef U1180779_LBVH_CONCRETE_CUH
#define U1180779_LBVH_CONCRETE_CUH

#include "lbvh/lbvh.cuh"
#include "unifiedObject.hpp"

struct aabb_getter
{
    // return aabb<float> if your object uses float.
    // if you chose double, return aabb<double>.
    __device__
        lbvh::aabb<float> operator()(const unifiedObject& f) const noexcept
    {
        // calculate aabb of object ...
        const float r = f.r;
        lbvh::aabb<float> box;
        box.upper = make_float4(f.x + r, f.y + r, f.z + r, 0.0f);
        box.lower = make_float4(f.x - r, f.y - r, f.z - r, 0.0f);
        return box;
    }
};

using bvh = lbvh::bvh<float, unifiedObject, aabb_getter>;
using bvhDevice = lbvh::bvh_device<float, unifiedObject>;

using bvhNode = lbvh::detail::node;
using bvhAABB = lbvh::aabb<float>;
using bvhNodeIdx = uint32_t;

#endif
