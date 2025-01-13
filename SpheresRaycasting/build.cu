// Copyright (c) 2022-2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "bvh.h"
#include "cub_helper.h"
#include "timer.hpp"

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ unsigned int expand_bits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// Calculates a 30-bit Morton code for the given 3D point located
/// within the unit cube [0,1].
__forceinline__ __device__ unsigned int morton_3d(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__global__ void assign_morton(bvh bvh)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= bvh.md_objects.count) 
        return;

    float3 pos = make_float3(
        bvh.md_objects.x[thread_id], 
        bvh.md_objects.y[thread_id], 
        bvh.md_objects.z[thread_id]);

    // normalize position
    float3 norm = (pos - bvh.md_objects.aabbMin) / bvh.md_objects.aabbRange;
    norm = max(min(norm, 1.0f), 1.0f);

    // obtain and set morton code based on normalized position
    bvh.md_sortObject.keysIn[thread_id] = morton_3d(norm.x, norm.y, norm.z);
    bvh.md_sortObject.indexIn[thread_id] = thread_id;
}

// todo this kernel is pretty small, can it be combined with another?
__global__ void leaf_nodes(int* sorted_object_ids, int num_objects, bvh bvh)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= num_objects) 
        return;

    // no need to set parent to nullptr, each child will have a parent
    bvh.leaf_nodes[thread_id].object_id = thread_id; // sorted_object_ids[thread_id]; // thread id?
    // needed to recognize that this node is a leaf
    bvh.leaf_nodes[thread_id].child_a = nullptr;

    // need to set for internal node parent to nullptr, for testing later
    // there is one less internal node than leaf node, test for that
    if (thread_id >= num_objects - 1) 
        return;
    bvh.internal_nodes[thread_id].parent = nullptr;
}

__forceinline__ __device__ int delta(int a, int b, unsigned int n, unsigned int* c, unsigned int ka)
{
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if (b < 0 || b > n - 1) return -1;
    unsigned int kb = c[b];
    if (ka == kb) {
        // if keys are equal, use id as fallback
        // (+32 because they have the same morton code)
        return 32 + __clz((unsigned int)a ^ (unsigned int)b);
    }
    // clz = count leading zeros
    return __clz(ka ^ kb);
}

__forceinline__ __device__ int2 determine_range(
    unsigned int* sorted_morton_codes, unsigned int n, int i)
{
    unsigned int* c = sorted_morton_codes;
    unsigned int ki = c[i]; // key of i

    // determine direction of the range (+1 or -1)
    const int delta_l = delta(i, i - 1, n, c, ki);
    const int delta_r = delta(i, i + 1, n, c, ki);

    int d; // direction
    int delta_min; // min of delta_r and delta_l
    if (delta_r < delta_l) {
        d = -1;
        delta_min = delta_r;
    }
    else {
        d = 1;
        delta_min = delta_l;
    }

    // compute upper bound of the length of the range
    unsigned int l_max = 2;
    while (delta(i, i + l_max * d, n, c, ki) > delta_min) {
        l_max <<= 1;
    }

    // find other end using binary search
    unsigned int l = 0;
    for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * d, n, c, ki) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d;

    // ensure i <= j
    return i < j ? make_int2(i, j) : make_int2(j, i);
}

__forceinline__ __device__ int find_split(
    unsigned int* sorted_morton_codes, int first, int last, unsigned int n)
{
    const unsigned int first_code = sorted_morton_codes[first];

    // calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic

    const int common_prefix =
        delta(first, last, n, sorted_morton_codes, first_code);

    // use binary search to find where the next bit differs
    // specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one

    int split = first; // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        const int new_split = split + step; // proposed new position

        if (new_split < last) {
            const int split_prefix = delta(
                first, new_split, n, sorted_morton_codes, first_code);
            if (split_prefix > common_prefix) {
                split = new_split; // accept proposal
            }
        }
    } while (step > 1);

    return split;
}

__global__ void internal_nodes(
    unsigned int* sorted_morton_codes, int* sorted_object_ids,
    int num_objects, bvh_node* d_leaf_nodes, bvh_node* d_internal_nodes)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // notice the -1, we want i in range [0, num_objects - 2]
    if (thread_id >= num_objects - 1) 
        return;

    // find out which range of objects the node corresponds to
    const int2 range = determine_range(
        sorted_morton_codes, num_objects, thread_id);

    // determine where to split the range
    const int split = find_split(
        sorted_morton_codes, range.x, range.y, num_objects);

    // select child a
    bvh_node* child_a;
    if (split == range.x) {
        child_a = &d_leaf_nodes[split];
    }
    else {
        child_a = &d_internal_nodes[split];
    }

    // select child b
    bvh_node* child_b;
    if (split + 1 == range.y) {
        child_b = &d_leaf_nodes[split + 1];
    }
    else {
        child_b = &d_internal_nodes[split + 1];
    }

    // record parent-child relationships
    d_internal_nodes[thread_id].child_a = child_a;
    d_internal_nodes[thread_id].child_b = child_b;
    d_internal_nodes[thread_id].visited = 0;
    child_a->parent = &d_internal_nodes[thread_id];
    child_b->parent = &d_internal_nodes[thread_id];
}

__global__ void set_aabb(bvh bvh)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= bvh.md_objects.count) 
        return;

    const unsigned int object_id = bvh.leaf_nodes[thread_id].object_id;

    // set bounding box of leaf node
    float r = bvh.md_objects.r[object_id];
    bvh.leaf_nodes[thread_id].min = make_float3(bvh.md_objects.x[object_id] - r, bvh.md_objects.y[object_id] - r, bvh.md_objects.z[object_id] - r);
    bvh.leaf_nodes[thread_id].max = make_float3(bvh.md_objects.x[object_id] + r, bvh.md_objects.y[object_id] + r, bvh.md_objects.z[object_id] + r);

    // recursively set tree bounding boxes
    // {current_node} is always an internal node (since it is parent of another)
    bvh_node* current_node = bvh.leaf_nodes[thread_id].parent;
    while (true) {
        // we have reached the parent of the root node: terminate
        if (current_node == nullptr) 
            break;

        // we have reached an inner node: check whether the node was visited
        unsigned int visited = atomicAdd(&current_node->visited, 1);

        // this is the first thread entering: terminate
        if (visited == 0) 
            break;

        // this is the second thread entering, we know that our sibling has
        // reached the current node and terminated,
        // and hence the sibling bounding box is correct

        // set running bounding box to be the union of bounding boxes
        current_node->min = fminf(
            current_node->child_a->min, current_node->child_b->min);
        current_node->max = fmaxf(
            current_node->child_a->max, current_node->child_b->max);

        // continue traversal
        current_node = current_node->parent;
    }
}

void bvh::build()
{
    // must have at least two triangles. we cannot build a bvh for zero
    // triangles, and a bvh of one triangle has no internal nodes
    // which requires special handling which we forgo
    if (md_objects.count <= 1) {
        fprintf(stderr, "too few objects in scene: %d", md_objects.count);
        abort();
    }

    std::cout << "\n\n\nTIMER time" << std::endl;

    timer t;
    t.start();
    dim3 block_count(md_objects.count / BLOCK_SIZE + 1);
    dim3 block_size(BLOCK_SIZE);

    assign_morton << <block_count, block_size >> > (*this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    t.stop("assign_morton");

    //t.start();
    //// sort is stable (https://groups.google.com/g/cub-users/c/1iXn3sVMEuA)
    //radix_sort(md_objects.count, 
    //    md_sortObject.keysIn, md_sortObject.keysOut, 
    //    md_sortObject.indexIn, md_sortObject.indexOut);
    //xcudaDeviceSynchronize();
    //xcudaGetLastError();
    //t.stop("radix_sort");

    t.start();
    md_sortObject.sort();
    t.stop("radix_sort");

    t.start();
    // construct leaf nodes
    ::leaf_nodes<<<block_count, block_size>>>(md_sortObject.indexOut, md_objects.count, *this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    t.stop("leaf_nodes");

    t.start();
    // construct internal nodes
    ::internal_nodes<<<block_count, block_size>>>(md_sortObject.keysOut, md_sortObject.indexOut, md_objects.count, leaf_nodes, internal_nodes);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    t.stop("internal_nodes");

    t.start();
    // calculate bounding boxes by walking the hierarchy toward the root
    set_aabb<<<block_count, block_size>>>(*this);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
    t.stop("set_aabb");
}
