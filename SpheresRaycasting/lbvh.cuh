
#ifndef U1180779_LBVH_H
#define U1180779_LBVH_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sortObject.cuh"
#include "unifiedObjects.hpp"

/* Linear bounding volume hierarchy (LBVH) */
/*
    sources:
    https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    https://luebke.us/publications/eg09.pdf
*/

/* aabb - axis aligned bounding box */

/* lbvh tree node */
class lbvhNode 
{
public:
    float3 aabbMin; /* aabb min vertex */
    float3 aabbMax; /* aabb box max vertex */

    /* stores one of:
    * index of left child, then right child = left child + 1
    * index of the first object in the bounding box
    depending on the count (number of objects)

    the number of objects in the bounding box for indermediate node is 0.
    Only the leaf nodes contain objects */
    int leftFirst;

    /* if node is leaft then is equal to number 
    of objects in the bounding box, 0 otherwise */
    int count; 
};

/* the coordinates (and r) are scaled to [0, 1] range */
struct mortonCodeData {
    int n;
    int bitsPerAxis;

    int* keys;
    float* x;
    float* y;
    float* z;
    float* r;
};

__global__ void computeMortonCode(mortonCodeData data);

class lbvh 
{
public:
    /* n - number of objects in scene */
    lbvh(unifiedObjects objects);
    ~lbvh();

    void sortByMortonCode();
    void construct();

    sortObject m_sortObject;
    lbvhNode* nodes; /* preallocated node memory [2*n - 1], node 0 is root */
    unifiedObjects m_objects; /* sorted objects (after constructing the tree) */
    const int m_n; /* number of objects in scene */
};

#endif

lbvh::lbvh(unifiedObjects objects) 
    : m_n(objects.count), m_sortObject(objects), m_objects(objects)
{
    nodes = new lbvhNode[m_n * 2 - 1];
}

lbvh::~lbvh()
{
    delete[] nodes;
}

void lbvh::sortByMortonCode()
{
    mortonCodeData data;
    data.keys = m_sortObject.keys;
    data.x = m_objects.x;
    data.y = m_objects.y;
    data.z = m_objects.z;
    data.r = m_objects.r;
    data.n = m_n;

    dim3 blocks = dim3(m_n / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);
    computeMortonCode<<<blocks, threads>>>(data);
    xcudaDeviceSynchronize();
    xcudaGetLastError();

    m_sortObject.sort();
}

void lbvh::construct()
{

}

__global__ void computeMortonCode(mortonCodeData data)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= data.n)
        return;

    int mortonX = (int)(data.x[i] * (1 << data.bitsPerAxis));
    int mortonY = (int)(data.z[i] * (1 << data.bitsPerAxis));
    int mortonZ = (int)(data.y[i] * (1 << data.bitsPerAxis));

    /* key is morton code */
    int mortonCode = 0;
    for (int i = 0; i < data.bitsPerAxis; ++i) {
        mortonCode |= ((mortonX >> i) & 1) << (3 * i);
        mortonCode |= ((mortonY >> i) & 1) << (3 * i + 1);
        mortonCode |= ((mortonZ >> i) & 1) << (3 * i + 2);
    }
    data.keys[i] = mortonCode;
}
