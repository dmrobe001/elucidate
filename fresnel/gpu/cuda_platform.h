// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef CUDA_PLATFORM_H_
#define CUDA_PLATFORM_H_

#include "common/ColorMath.h"
#include "common/VectorMath.h"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

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
//! Geometry id reported when a ray hits nothing
static const unsigned int INVALID_GEOMETRY_ID = 0xffffffff;

//! Result of a ray query against the Scene
/*! Embree carries the fresnel specific hit attributes on the ray query context (see
   cpu/embree_platform.h). There is no equivalent hand-off point here, so the traversal fills out
   this structure directly. It carries the same information:

    - *t*: Distance along the ray to the hit
    - *d*: The distance to the nearest edge, provided by intersection routines
    - *shading_color*: The color of the primitive (or primitive subunit), provided by intersection
      routines
    - *Ng*: The *geometric* normal, pointing out of the primitive whichever side was struck
    - *geom_id*: The geometry that was hit, used to look up materials in the Scene
*/
struct HitInfo
    {
    DEVICE HitInfo() : t(0.0f), d(0.0f), Ng(0, 0, 0), geom_id(INVALID_GEOMETRY_ID) { }

    //! True when the ray hit geometry
    DEVICE bool hit() const
        {
        return geom_id != INVALID_GEOMETRY_ID;
        }

    float t; //!< Distance along the ray to the hit point
    float d; //!< Distance to the nearest edge
    RGB<float> shading_color; //!< shading color determined by which primitive the ray hits
                              //!< (or where on the primitive)
    vec3<float> Ng; //!< Geometric normal at the hit point
    unsigned int geom_id; //!< Geometry that was hit
    };

//! Throw a std::runtime_error describing a failed CUDA call
inline void check_cuda_error(cudaError_t error, const char* file, unsigned int line)
    {
    if (error != cudaSuccess)
        {
        std::ostringstream s;
        s << "CUDA: " << cudaGetErrorString(error) << " (" << file << ":" << line << ")";
        throw std::runtime_error(s.str());
        }
    }

    } // namespace gpu
    } // namespace fresnel

//! Check the return value of a CUDA API call and throw on failure
#define CUDA_CHECK(call) ::fresnel::gpu::check_cuda_error((call), __FILE__, __LINE__)

//! Check for an error left behind by an asynchronous launch
#define CUDA_CHECK_LAUNCH()                                                            \
        {                                                                              \
        ::fresnel::gpu::check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);      \
        ::fresnel::gpu::check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__); \
        }

#undef DEVICE

#endif
