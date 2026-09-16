// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "BVHBuild.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

namespace fresnel
    {
namespace gpu
    {
namespace kernel
    {
//! Number of threads in a build kernel block
static const unsigned int build_block_size = 128;

//! Flatten every geometry's primitives into one array
__global__ void fill_primitives(PrimitiveRef* prims,
                                const unsigned int* offsets,
                                unsigned int n_geometry,
                                unsigned int n_prims)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_prims)
        return;

    const unsigned int g = find_geometry(offsets, n_geometry, i);
    prims[i].geom_id = g;
    prims[i].prim_id = i - offsets[g];
    }

//! Bound every primitive
__global__ void compute_bounds(AABB* aabb,
                               const PrimitiveRef* prims,
                               const GeometryData* geometry,
                               unsigned int n_prims)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_prims)
        return;

    const PrimitiveRef ref = prims[i];
    geometry_bounds(geometry[ref.geom_id], ref.prim_id, aabb[i].lower, aabb[i].upper);
    }

//! Morton code the centroid of every primitive and seed the sort permutation
__global__ void compute_morton(unsigned int* morton,
                               unsigned int* index,
                               const AABB* aabb,
                               vec3<float> scene_lower,
                               vec3<float> scene_extent,
                               unsigned int n_prims)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_prims)
        return;

    const vec3<float> centroid = (aabb[i].lower + aabb[i].upper) * 0.5f;
    const vec3<float> unit((centroid.x - scene_lower.x) / scene_extent.x,
                           (centroid.y - scene_lower.y) / scene_extent.y,
                           (centroid.z - scene_lower.z) / scene_extent.z);

    morton[i] = morton_code(unit);
    index[i] = i;
    }

//! Write the leaf nodes in Morton order
__global__ void write_leaves(BVHNode* nodes,
                             PrimitiveRef* prims,
                             const PrimitiveRef* unsorted_prims,
                             const AABB* aabb,
                             const unsigned int* index,
                             unsigned int n_prims)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_prims)
        return;

    const unsigned int source = index[i];
    prims[i] = unsorted_prims[source];

    BVHNode& node = nodes[n_prims - 1 + i];
    node.lower = aabb[source].lower;
    node.upper = aabb[source].upper;
    node.left = -1;
    node.right = (int)i;
    }

//! Link the sorted leaves into a binary radix tree
__global__ void
build_radix_tree(BVHNode* nodes, int* parent, const unsigned int* morton, unsigned int n_prims)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_prims - 1)
        return;

    int left, right;
    radix_tree_node(morton, (int)n_prims, (int)idx, left, right);

    nodes[idx].left = left;
    nodes[idx].right = right;
    parent[left] = (int)idx;
    parent[right] = (int)idx;
    }

//! Fit bounding boxes to the internal nodes, from the leaves up
/*! Each leaf walks toward the root. The first thread to reach a node stops; the second finds both
   children finished and merges their boxes, so every node is visited exactly once.
*/
__global__ void fit_bounds(BVHNode* nodes, int* visit, const int* parent, unsigned int n_prims)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_prims)
        return;

    int node = parent[n_prims - 1 + i];

    while (node >= 0)
        {
        __threadfence();

        if (atomicAdd(&visit[node], 1) == 0)
            return;

        AABB box;
        box.lower = nodes[nodes[node].left].lower;
        box.upper = nodes[nodes[node].left].upper;

        AABB other;
        other.lower = nodes[nodes[node].right].lower;
        other.upper = nodes[nodes[node].right].upper;

        box = merge(box, other);
        nodes[node].lower = box.lower;
        nodes[node].upper = box.upper;

        node = parent[node];
        }
    }

//! Merge two bounding boxes, for thrust::reduce
struct MergeAABB
    {
    __host__ __device__ AABB operator()(const AABB& a, const AABB& b) const
        {
        return merge(a, b);
        }
    };

    } // namespace kernel

/*! \param geometry Geometry descriptions, indexed by geometry id
    \param offsets Prefix sum of the primitive counts, n_geometry + 1 entries
    \param n_geometry Number of geometries
*/
void BVH::build(const GeometryData* geometry, const unsigned int* offsets, unsigned int n_geometry)
    {
    // offsets is managed memory written by the host, so nothing has to be copied back
    m_n_prims = (n_geometry == 0) ? 0 : offsets[n_geometry];

    if (m_n_prims == 0)
        return;

    const unsigned int n = m_n_prims;
    const unsigned int n_nodes = (n == 1) ? 1 : (2 * n - 1);

    m_unsorted_prims.resize(n);
    m_prims.resize(n);
    m_aabb.resize(n);
    m_morton.resize(n);
    m_index.resize(n);
    m_nodes.resize(n_nodes);
    m_parent.resize(n_nodes);
    m_visit.resize(n_nodes);

    // -1 is the root's parent, and every counter starts at zero on each build
    CUDA_CHECK(cudaMemset(m_parent.get(), 0xff, sizeof(int) * n_nodes));
    CUDA_CHECK(cudaMemset(m_visit.get(), 0, sizeof(int) * n_nodes));

    const unsigned int n_blocks = (n + kernel::build_block_size - 1) / kernel::build_block_size;

    kernel::fill_primitives<<<n_blocks, kernel::build_block_size>>>(m_unsorted_prims.get(),
                                                                    offsets,
                                                                    n_geometry,
                                                                    n);
    CUDA_CHECK_LAUNCH();

    kernel::compute_bounds<<<n_blocks, kernel::build_block_size>>>(m_aabb.get(),
                                                                   m_unsorted_prims.get(),
                                                                   geometry,
                                                                   n);
    CUDA_CHECK_LAUNCH();

    // bound the whole scene so that the centroids can be quantized
    AABB init;
    init.lower = vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
    init.upper = vec3<float>(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    thrust::device_ptr<AABB> aabb_begin(m_aabb.get());
    const AABB scene = thrust::reduce(aabb_begin, aabb_begin + n, init, kernel::MergeAABB());

    // a scene with no extent in some direction still has to produce a usable Morton code
    vec3<float> extent = scene.upper - scene.lower;
    if (!(extent.x > 0.0f))
        extent.x = 1.0f;
    if (!(extent.y > 0.0f))
        extent.y = 1.0f;
    if (!(extent.z > 0.0f))
        extent.z = 1.0f;

    kernel::compute_morton<<<n_blocks, kernel::build_block_size>>>(m_morton.get(),
                                                                   m_index.get(),
                                                                   m_aabb.get(),
                                                                   scene.lower,
                                                                   extent,
                                                                   n);
    CUDA_CHECK_LAUNCH();

    thrust::device_ptr<unsigned int> morton_begin(m_morton.get());
    thrust::device_ptr<unsigned int> index_begin(m_index.get());
    thrust::sort_by_key(morton_begin, morton_begin + n, index_begin);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel::write_leaves<<<n_blocks, kernel::build_block_size>>>(m_nodes.get(),
                                                                 m_prims.get(),
                                                                 m_unsorted_prims.get(),
                                                                 m_aabb.get(),
                                                                 m_index.get(),
                                                                 n);
    CUDA_CHECK_LAUNCH();

    // a single primitive is its own root: there are no internal nodes to link or fit
    if (n == 1)
        return;

    kernel::build_radix_tree<<<n_blocks, kernel::build_block_size>>>(m_nodes.get(),
                                                                     m_parent.get(),
                                                                     m_morton.get(),
                                                                     n);
    CUDA_CHECK_LAUNCH();

    kernel::fit_bounds<<<n_blocks, kernel::build_block_size>>>(m_nodes.get(),
                                                               m_visit.get(),
                                                               m_parent.get(),
                                                               n);
    CUDA_CHECK_LAUNCH();
    }

    } // namespace gpu
    } // namespace fresnel
