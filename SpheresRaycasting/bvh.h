
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

static void traverse(bvh_node* node) 
{
    if (!node)
        return;
    traverse(node->child_a);
    if (!node->child_a) 
    {
        std::cout << "objectId = " << node->object_id << std::endl;
        std::cout << "\tmin = (" << node->min.x << ", " << node->min.y << ", " << node->min.z << ")" << std::endl;
        std::cout << "\tmax = (" << node->max.x << ", " << node->max.y << ", " << node->max.z << ")" << std::endl;
    }
    traverse(node->child_b);
}

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

    void toHost() 
    {
        bvh_node* h_leaf = new bvh_node[md_objects.count];
        bvh_node* h_internal = new bvh_node[md_objects.count - 1];
        cudaMemcpy(h_leaf, leaf_nodes, sizeof(bvh_node) * md_objects.count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_internal, internal_nodes, sizeof(bvh_node) * (md_objects.count - 1), cudaMemcpyDeviceToHost);

        void* helper = new void*;
        cudaMemcpy(helper, internal_nodes, sizeof(void*), cudaMemcpyDeviceToHost);

        for(int i = 0; i < md_objects.count - 1; ++i) 
        {
            bvh_node* node = &h_internal[i];
            std::cout << "internal" << std::endl;
            std::cout << "\tchild_a = " << node->child_a << std::endl;
            std::cout << "\tchild_b = " << node->child_b << std::endl;
            std::cout << "\tmin = (" << node->min.x << ", " << node->min.y << ", " << node->min.z << ")" << std::endl;
            std::cout << "\tmax = (" << node->max.x << ", " << node->max.y << ", " << node->max.z << ")" << std::endl;

            if (node->child_a > h_leaf)
            {
                node->child_a = node->child_a - h_leaf + h_leaf;
            }
            else 
            {
                node->child_a = node->child_a - internal_nodes + h_internal;
            }


            if (node->child_b > h_leaf)
            {
                node->child_b = node->child_b - h_leaf + h_leaf;
            }
            else
            {
                node->child_b = node->child_b - internal_nodes + h_internal;
            }
        }

        std::cout << "\ninternal nodes: \n";
        traverse(&h_internal[0]);

        delete[] h_leaf;
        delete[] h_internal;
    }
};



