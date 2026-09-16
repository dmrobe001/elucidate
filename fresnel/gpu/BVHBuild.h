// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef BVH_BUILD_H_
#define BVH_BUILD_H_

#include "BVH.h"

#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#include <intrin.h>
#endif

/*! \file BVHBuild.h
    \brief The per-thread steps of the linear BVH build.

    BVH.cu reduces to a thread index and a store around these functions. Each one is a pure function
   of the Morton codes and the primitive bounds, which is what makes the build parallel in the first
   place, and what lets it be checked without a device.
*/

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
//! Count the leading zeros of a 32 bit word
DEVICE inline int clz32(unsigned int v)
    {
#ifdef __CUDA_ARCH__
    return __clz((int)v);
#elif defined(_MSC_VER)
    unsigned long index;
    return _BitScanReverse(&index, v) ? (31 - (int)index) : 32;
#else
    return (v == 0) ? 32 : __builtin_clz(v);
#endif
    }

//! Merge two bounding boxes
DEVICE inline AABB merge(const AABB& a, const AABB& b)
    {
    AABB r;
    r.lower = vec3<float>(fminf(a.lower.x, b.lower.x),
                          fminf(a.lower.y, b.lower.y),
                          fminf(a.lower.z, b.lower.z));
    r.upper = vec3<float>(fmaxf(a.upper.x, b.upper.x),
                          fmaxf(a.upper.y, b.upper.y),
                          fmaxf(a.upper.z, b.upper.z));
    return r;
    }

//! Locate the geometry a flattened primitive index belongs to
/*! \param offsets Prefix sum of the primitive counts
    \param n_geometry Number of geometries
    \param i Flattened primitive index

    \returns The geometry id, i.e. the largest g with offsets[g] <= i
*/
DEVICE inline unsigned int
find_geometry(const unsigned int* offsets, unsigned int n_geometry, unsigned int i)
    {
    unsigned int lo = 0;
    unsigned int hi = n_geometry;

    while (hi - lo > 1)
        {
        const unsigned int mid = (lo + hi) / 2;
        if (offsets[mid] <= i)
            lo = mid;
        else
            hi = mid;
        }

    return lo;
    }

//! Spread the low 10 bits of \a v out so that two zero bits sit between each
DEVICE inline unsigned int expand_bits(unsigned int v)
    {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
    }

//! 30 bit Morton code of a point in the unit cube
DEVICE inline unsigned int morton_code(vec3<float> p)
    {
    p.x = fminf(fmaxf(p.x * 1024.0f, 0.0f), 1023.0f);
    p.y = fminf(fmaxf(p.y * 1024.0f, 0.0f), 1023.0f);
    p.z = fminf(fmaxf(p.z * 1024.0f, 0.0f), 1023.0f);

    return expand_bits((unsigned int)p.x) * 4 + expand_bits((unsigned int)p.y) * 2
           + expand_bits((unsigned int)p.z);
    }

//! Length of the common prefix of two Morton codes
/*! Duplicate codes fall back to the primitive indices, which keeps the radix tree well defined when
   several primitives share a cell.
*/
DEVICE inline int delta(const unsigned int* morton, int n, int i, int j)
    {
    if (j < 0 || j >= n)
        return -1;

    const unsigned int mi = morton[i];
    const unsigned int mj = morton[j];

    if (mi == mj)
        return 32 + clz32((unsigned int)i ^ (unsigned int)j);

    return clz32(mi ^ mj);
    }

//! Find the children of one internal node of the binary radix tree
/*! \param morton Sorted Morton codes
    \param n Number of leaves
    \param i Index of the internal node, in [0, n - 1)
    \param left [out] Index of the left child in the node array
    \param right [out] Index of the right child in the node array

    Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (2012).
   Internal node \a i covers a range of leaves determined entirely from the Morton codes, so every
   node is built independently. Children with an index of n - 1 or greater are leaves.
*/
DEVICE inline void radix_tree_node(const unsigned int* morton, int n, int i, int& left, int& right)
    {
    // determine the direction of the range
    const int d = (delta(morton, n, i, i + 1) - delta(morton, n, i, i - 1)) >= 0 ? 1 : -1;

    // find an upper bound on the length of the range
    const int delta_min = delta(morton, n, i, i - d);
    int l_max = 2;
    while (delta(morton, n, i, i + l_max * d) > delta_min)
        l_max *= 2;

    // binary search for the other end of the range
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2)
        {
        if (delta(morton, n, i, i + (l + t) * d) > delta_min)
            l += t;
        }
    const int j = i + l * d;

    // binary search for the split position within the range
    const int delta_node = delta(morton, n, i, j);
    int s = 0;
    int t = l;
    do
        {
        t = (t + 1) / 2;
        if (delta(morton, n, i, i + (s + t) * d) > delta_node)
            s += t;
        } while (t > 1);

    const int split = i + s * d + ((d < 0) ? -1 : 0);

    const int first = (i < j) ? i : j;
    const int last = (i < j) ? j : i;

    left = (split == first) ? (n - 1 + split) : split;
    right = (split + 1 == last) ? (n - 1 + split + 1) : (split + 1);
    }

    } // namespace gpu
    } // namespace fresnel

#undef DEVICE

#endif
