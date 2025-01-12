
#ifndef U1180779_LBVH_H
#define U1180779_LBVH_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sortObject.cuh"
#include "unifiedObjects.cuh"

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

   
    int leftChild;  /* index of left child or 0 */
    int rightChild;  /* index of right child or 0 */
    int object;  /* index of object */
};

/* the coordinates (and r) are scaled to [0, 1] range */
struct mortonCodeData {
    int n;
    int bitsPerAxis = 6;

    int* keys;
    float* x;
    float* y;
    float* z;
};

__global__ void computeMortonCode(mortonCodeData data);
__device__ int generateHierarchy(int* sortedMortonCodes, int first, int last, int* nodeNr, lbvhNode* nodes);
__device__ int findSplit(int* sortedMortonCodes, int first, int last);

class lbvh 
{
public:
    /* n - number of objects in scene */
    lbvh(unifiedObjects objects);
    ~lbvh();

    void sortByMortonCode();
    void construct();

    mortonCodeData m_mortonData;
    sortObject m_sortObject;
    lbvhNode* m_nodes; /* preallocated node memory [numer of Morton codes], node 0 is root */
    unifiedObjects m_objects; /* sorted objects (after constructing the tree) */
    const int m_n; /* number of objects in scene */
    int* m_nodeNr;
};

#endif



