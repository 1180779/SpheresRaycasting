
#include "lbvh.cuh"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "timer.hpp"

lbvh::lbvh(unifiedObjects objects)
    : m_n(objects.count), m_objects(objects)
{
    m_sortObject.malloc(objects);
    m_mortonData.n = objects.count;
    xcudaMalloc(&m_nodes, sizeof(lbvhNode) * (1024 * 256));
    xcudaMalloc(&m_nodeNr, sizeof(int));
}

lbvh::~lbvh()
{
    m_sortObject.free();
    xcudaFree(m_nodes);
    xcudaFree(m_nodeNr);
}

void lbvh::sortByMortonCode()
{
    float3 minCoords;
    float3 maxCoords;

    timer t;
    t.start();

    minCoords.x = *thrust::min_element(
        thrust::device_ptr<float>(m_objects.x),
        thrust::device_ptr<float>(m_objects.x + m_objects.count));
    minCoords.y = *thrust::min_element(
        thrust::device_ptr<float>(m_objects.y),
        thrust::device_ptr<float>(m_objects.y + m_objects.count));
    minCoords.z = *thrust::min_element(
        thrust::device_ptr<float>(m_objects.z),
        thrust::device_ptr<float>(m_objects.z + m_objects.count));

    maxCoords.x = *thrust::max_element(
        thrust::device_ptr<float>(m_objects.x),
        thrust::device_ptr<float>(m_objects.x + m_objects.count));
    maxCoords.y = *thrust::max_element(
        thrust::device_ptr<float>(m_objects.y),
        thrust::device_ptr<float>(m_objects.y + m_objects.count));
    maxCoords.z = *thrust::max_element(
        thrust::device_ptr<float>(m_objects.z),
        thrust::device_ptr<float>(m_objects.z + m_objects.count));
    t.stop("find min max of scene");

    float3 range;
    range.x = maxCoords.x - minCoords.x;

    // clear the tree
    int source = 0;
    cudaMemcpy(m_nodeNr, &source, sizeof(int), cudaMemcpyHostToDevice);

    m_mortonData.keys = m_sortObject.keysIn;

    t.start();
    dim3 blocks = dim3(m_n / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);
    computeMortonCodeKernel<<<blocks, threads>>>(m_mortonData, m_objects, minCoords, range);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    t.stop("computeMortonCodeKernel");

    t.start();
    m_sortObject.sort();
    t.stop("sort by morton codes");
}

void lbvh::construct()
{
    generateHierarchyRunner << <1, 1 >> > (m_mortonData.keys, 0, m_n - 1, m_nodeNr, m_nodes);
    xcudaDeviceSynchronize();
    xcudaGetLastError();

    // TODO: create bounding boxes from lowest level up
}

__device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ unsigned int morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__global__ void computeMortonCodeKernel(mortonCodeData codes, unifiedObjects objects, float3 min, float3 range)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= codes.n)
        return;

    float x = (objects.x[i] - min.x) / range.x;
    float y = (objects.y[i] - min.y) / range.y;
    float z = (objects.z[i] - min.z) / range.z;
    codes.keys[i] = morton3D(x, y, z);
}

__global__ void generateHierarchyRunner(unsigned int* sortedMortonCodes, int first, int last, int* nodeNr, lbvhNode* nodes) 
{
    generateHierarchy(sortedMortonCodes, first, last, nodeNr, nodes);
}

__device__ int generateHierarchy(unsigned int* sortedMortonCodes, int first, int last, int* nodeNr, lbvhNode* nodes)
{
    if (first == last)
    {
        int i = atomicAdd(nodeNr, 1);
        i = i - 1;
        nodes[i].leftChild = 0;
        nodes[i].rightChild = 0;
        nodes[i].object = first;
        return i;
    }

    int i = atomicAdd(nodeNr, 1);
    i = i - 1;
    // determine where to split range

    int split = findSplit(sortedMortonCodes, first, last);

    // Process the resulting sub-ranges recursively.

    int childA = generateHierarchy(sortedMortonCodes, first, split, nodeNr, nodes);
    int childB = generateHierarchy(sortedMortonCodes, split + 1, last, nodeNr, nodes);

    nodes[i].leftChild = childA;
    nodes[i].rightChild = childB;
    nodes[i].object = 0;
    return i;
}

// copied from
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__device__ int findSplit(unsigned int* sortedMortonCodes, int first, int last)
{
    // Identical Morton codes => split the range in the middle.

    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}
