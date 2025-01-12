
#include "lbvh.cuh"

lbvh::lbvh(unifiedObjects objects)
    : m_n(objects.count), m_objects(objects)
{
    m_sortObject.malloc(objects);
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
    // clear the tree
    int source = 0;
    cudaMemcpy(m_nodeNr, &source, sizeof(int), cudaMemcpyHostToDevice);

    m_mortonData;
    m_mortonData.keys = m_sortObject.keys;
    m_mortonData.x = m_objects.x;
    m_mortonData.y = m_objects.y;
    m_mortonData.z = m_objects.z;
    m_mortonData.n = m_n;

    dim3 blocks = dim3(m_n / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);
    computeMortonCode << <blocks, threads >> > (m_mortonData);
    xcudaDeviceSynchronize();
    xcudaGetLastError();

    m_sortObject.sort();
}

void lbvh::construct()
{
    //generateHierarchyRunner << <1, 1 >> > (m_mortonData.keys, 0, m_n - 1, m_nodeNr, m_nodes);
    //xcudaDeviceSynchronize();
    //xcudaGetLastError();

    // TODO: create bounding boxes from lowest level up
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

__device__ int generateHierarchy(int* sortedMortonCodes, int first, int last, int* nodeNr, lbvhNode* nodes)
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
__device__ int findSplit(int* sortedMortonCodes, int first, int last)
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
