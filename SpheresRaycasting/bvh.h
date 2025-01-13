
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "unifiedObjects.cuh"
#include "sortObject.cuh"
#include "cudaInline.h"

struct hit {
    float3 hitpoint;
    /// Whether a hit is recorded.
    int hit;
    float3 normal;
};

struct bvh_node {
    /// If nullptr, then this node is a leaf.
    bvh_node* child_a;
    bvh_node* child_b;
    bvh_node* parent;
    /** Bounding box. */
    float3 min;
    float3 max;

    /// If leaf, holds the id of the object.
    unsigned int object_id;
    /// If internal node, holds whether the node has been visited once
    /// while setting bounding boxes. The first thread (child) sets
    /// it equal to its own bounding box and continues up the tree.
    /// The second thread (child) sets it to their union and terminates.
    unsigned int visited;

    __forceinline__ __device__ bool is_leaf() const
    {
        return child_a == nullptr;
    }
};

struct bvh {
    /// Leaf nodes, one for every photon data element.
    bvh_node* leaf_nodes;
    /// Internal nodes, amount equal to #leaf nodes - 1.
    bvh_node* internal_nodes;
    /// Scene, but all arrays are allocated on device.
    unifiedObjects md_objects;

    sortObject md_sortObject;

    void build();

    void malloc(unifiedObjects dObjects)
    {
        md_objects = dObjects;
        md_sortObject.malloc(dObjects);

        // allocate BVH (n leaf nodes, n - 1 internal nodes)
        xcudaMalloc(&leaf_nodes, sizeof(bvh_node) * md_objects.count);
        xcudaMalloc(&internal_nodes, sizeof(bvh_node) * (md_objects.count - 1));
    }

    void free() 
    {
        md_sortObject.free();

        xcudaFree(leaf_nodes);
        xcudaFree(internal_nodes);
    }
};


