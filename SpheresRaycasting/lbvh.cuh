
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
    unsigned int* keys;
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z);

__global__ void computeMortonCodeKernel(mortonCodeData codes, unifiedObjects objects, float3 min, float3 range);

__global__ void generateHierarchyRunner(unsigned int* sortedMortonCodes, int first, int last, int* nodeNr, lbvhNode* nodes);
__device__ int generateHierarchy(unsigned int* sortedMortonCodes, int first, int last, int* nodeNr, lbvhNode* nodes);
__device__ int findSplit(unsigned int* sortedMortonCodes, int first, int last);

class lbvh
{
public:
    /* n - number of objects in scene */
    lbvh(unifiedObjects objects);
    ~lbvh();

    void normalizeCoords();
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
