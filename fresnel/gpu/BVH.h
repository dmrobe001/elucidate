// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef BVH_H_
#define BVH_H_

#include "Array.h"
#include "GeometryData.h"
#include "cuda_platform.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE
#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace fresnel
    {
namespace gpu
    {
//! Reference to one primitive somewhere in the Scene
/*! Every geometry's primitives are flattened into a single array and one BVH is built over all of
   them. The tracers only ever need \a geom_id, which is what materials and outline widths are
   indexed by.
*/
struct PrimitiveRef
    {
    unsigned int geom_id; //!< Geometry the primitive belongs to
    unsigned int prim_id; //!< Index of the primitive within that geometry
    };

//! One node of the BVH
/*! Nodes are stored in a single array. The first \a n_prims - 1 entries are internal nodes and the
   remainder are leaves, following Karras' layout for a linear BVH. A leaf is marked by a negative
   \a left; its \a right is then the slot in the sorted primitive array that it covers.

    A one primitive scene has no internal nodes, so node 0 is the leaf itself. Traversal always
   starts at node 0 either way.
*/
struct BVHNode
    {
    vec3<float> lower; //!< Lower corner of the bounding box
    vec3<float> upper; //!< Upper corner of the bounding box
    int left; //!< Index of the left child, or -1 for a leaf
    int right; //!< Index of the right child, or the primitive slot for a leaf
    };

//! Axis aligned bounding box of one primitive
struct AABB
    {
    vec3<float> lower; //!< Lower corner
    vec3<float> upper; //!< Upper corner
    };

//! Device visible view of a built BVH
struct BVHView
    {
    const BVHNode* nodes; //!< Node array
    const PrimitiveRef* prims; //!< Primitives in BVH order
    unsigned int n_prims; //!< Number of primitives
    };

//! Maximum BVH traversal stack depth
/*! Each level of the radix tree consumes at least one bit of the key it splits on: 30 bits of
   Morton code, and, where those are equal, 32 bits of primitive index. The tree is therefore at
   most 62 levels deep and the stack holds at most one entry per level above the current node.
*/
static const unsigned int BVH_STACK_SIZE = 64;

//! Slab test of a ray against an axis aligned box
/*! \param lower Lower corner of the box
    \param upper Upper corner of the box
    \param org Ray origin
    \param inv_dir Componentwise reciprocal of the ray direction
    \param t_near Near clipping distance
    \param t_far Far clipping distance

    \returns true when the ray passes through the box within (t_near, t_far)

    fminf and fmaxf return the non-NaN operand, which is what keeps a ray that lies exactly in the
   plane of a slab (0 * infinity) from dropping the box.
*/
DEVICE inline bool intersect_box(const vec3<float>& lower,
                                 const vec3<float>& upper,
                                 const vec3<float>& org,
                                 const vec3<float>& inv_dir,
                                 float t_near,
                                 float t_far)
    {
    const float tx0 = (lower.x - org.x) * inv_dir.x;
    const float tx1 = (upper.x - org.x) * inv_dir.x;
    const float ty0 = (lower.y - org.y) * inv_dir.y;
    const float ty1 = (upper.y - org.y) * inv_dir.y;
    const float tz0 = (lower.z - org.z) * inv_dir.z;
    const float tz1 = (upper.z - org.z) * inv_dir.z;

    float tmin = fmaxf(fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)), fmaxf(fminf(tz0, tz1), t_near));
    float tmax = fminf(fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)), fminf(fmaxf(tz0, tz1), t_far));

    return tmin <= tmax;
    }

//! Find the closest hit along a ray
/*! \param bvh The acceleration structure to traverse
    \param geometry Geometry descriptions, indexed by geometry id
    \param org Ray origin
    \param dir Ray direction (normalized)
    \param t_near Near clipping distance
    \param t_far Far clipping distance

    \returns The closest hit, or a HitInfo with geom_id == INVALID_GEOMETRY_ID when the ray hits
   nothing.

    This is the equivalent of rtcIntersect1 plus the fresnel extensions Embree carries on its ray
   query context. The traversal narrows \a t_far as it goes, which is what lets the convex
   polyhedron routine reproduce Embree's entry-plane-wins behaviour.
*/
DEVICE inline HitInfo bvh_intersect(const BVHView& bvh,
                                    const GeometryData* geometry,
                                    const vec3<float>& org,
                                    const vec3<float>& dir,
                                    float t_near,
                                    float t_far)
    {
    HitInfo best;

    if (bvh.n_prims == 0)
        return best;

    const vec3<float> inv_dir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    int stack[BVH_STACK_SIZE];
    unsigned int stack_size = 0;
    int node_index = 0;

    while (true)
        {
        const BVHNode node = bvh.nodes[node_index];

        if (node.left < 0)
            {
            // leaf: intersect the primitive it covers
            const PrimitiveRef ref = bvh.prims[node.right];
            const GeometryData& geom = geometry[ref.geom_id];

            if (geom.enabled)
                {
                HitInfo candidate;
                if (geometry_intersect(geom, ref.prim_id, org, dir, t_near, t_far, candidate))
                    {
                    candidate.geom_id = ref.geom_id;
                    best = candidate;
                    t_far = candidate.t;
                    }
                }
            }
        else
            {
            const BVHNode& left = bvh.nodes[node.left];
            const BVHNode& right = bvh.nodes[node.right];

            const bool hit_left
                = intersect_box(left.lower, left.upper, org, inv_dir, t_near, t_far);
            const bool hit_right
                = intersect_box(right.lower, right.upper, org, inv_dir, t_near, t_far);

            if (hit_left && hit_right)
                {
                if (stack_size < BVH_STACK_SIZE)
                    stack[stack_size++] = node.right;
                node_index = node.left;
                continue;
                }
            else if (hit_left)
                {
                node_index = node.left;
                continue;
                }
            else if (hit_right)
                {
                node_index = node.right;
                continue;
                }
            }

        if (stack_size == 0)
            break;

        node_index = stack[--stack_size];
        }

    return best;
    }

//! Linear bounding volume hierarchy over every primitive in a Scene
/*! Embree supplies the acceleration structure for the CPU backend; this is its counterpart. The
   tree is a Morton code LBVH (Karras 2012), built entirely on the device: the primitives are
   flattened, bounded, sorted by the Morton code of their centroid, linked into a radix tree, and
   then fitted with bounding boxes from the leaves up.

    Scene::commit() calls build() on every render, which is where the CPU backend calls
   rtcCommitScene(). The tree is rebuilt unconditionally rather than refitted, since a Scene has no
   way to know which buffers python wrote to between renders.
*/
class BVH
    {
    public:
    BVH() : m_n_prims(0) { }

    //! Rebuild the hierarchy
    /*! \param geometry Geometry descriptions, indexed by geometry id
        \param offsets Prefix sum of the primitive counts, n_geometry + 1 entries
        \param n_geometry Number of geometries
    */
    void build(const GeometryData* geometry, const unsigned int* offsets, unsigned int n_geometry);

    //! Get a device visible view of the hierarchy
    BVHView view() const
        {
        BVHView v;
        v.nodes = m_nodes.get();
        v.prims = m_prims.get();
        v.n_prims = m_n_prims;
        return v;
        }

    private:
    ManagedBuffer<BVHNode> m_nodes; //!< Internal nodes followed by leaves
    ManagedBuffer<PrimitiveRef> m_prims; //!< Primitives in Morton order
    ManagedBuffer<PrimitiveRef> m_unsorted_prims; //!< Primitives in geometry order
    ManagedBuffer<AABB> m_aabb; //!< Bounding box of each primitive, in geometry order
    ManagedBuffer<unsigned int> m_morton; //!< Morton code of each primitive
    ManagedBuffer<unsigned int> m_index; //!< Permutation produced by the Morton sort
    ManagedBuffer<int> m_parent; //!< Parent of each node, -1 at the root
    ManagedBuffer<int> m_visit; //!< Atomic arrival counter used by the bounding box fit
    unsigned int m_n_prims; //!< Number of primitives in the hierarchy
    };

    } // namespace gpu
    } // namespace fresnel

#undef DEVICE

#endif
