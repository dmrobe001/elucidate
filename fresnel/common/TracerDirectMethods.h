// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "ColorMath.h"
#include "Light.h"
#include "Material.h"
#include "VectorMath.h"

#ifndef __TRACER_DIRECT_METHODS_H__
#define __TRACER_DIRECT_METHODS_H__

/*! \file TracerDirectMethods.h
    \brief The per-hit body of the direct tracer.

    This is to tracer.Preview what TracerPathMethods.h is to tracer.Path: the shading itself is
   backend neutral, and only the traversal around it - rtcIntersect1() on the CPU, bvh_intersect()
   on the GPU - belongs to a backend.
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
//! Most interfaces the direct tracer follows one ray through
/*! A ray that refracts into a solid and then totally internally reflects can keep bouncing
    around inside it, and unlike the path tracer there is no Russian roulette here to end it.
    A ray that runs out of crossings shades the surface it is standing on as though it were
    opaque, which is a bounded and visible answer rather than a black pixel. Sixteen is deep
    enough to see through several sheets of glass, and to get in and back out of a solid that
    reflects internally a few times on the way.
*/
const unsigned int direct_tracer_max_crossings = 16;

//! Whether the direct tracer sees through a material
/*! \param m Material at the hit

    \returns True when a ray continues through the surface instead of shading it

    Any transmission at all makes a surface fully transparent here. tracer.Preview shades one
    hit against the lights and has no samples to split between an opaque lobe and a dielectric
    one, so it takes the simplest reading of \a spec_trans that still shows what is behind the
    surface. A solid material stays opaque whatever its \a spec_trans, matching the path
    tracer, where isSolid() likewise wins and turns the surface into an emitter.
*/
DEVICE inline bool direct_tracer_is_transparent(const Material& m)
    {
    return !m.isSolid() && m.spec_trans > 0.0f;
    }

//! Continue a ray through a transparent surface
/*! \param direction [output] Direction to continue the ray with
    \param tint [in,out] Color the pixel is multiplied by, scaled on getting through the surface
    \param m Material at the hit
    \param shading_color Color of the primitive
    \param n Normal, already flipped to the side the ray arrived from
    \param v Unit vector pointing back along the incoming ray
    \param backfacing True when the ray struck the inside of the surface

    The interface is smooth whatever the material's \a roughness, and nothing reflects off it:
    "transparent" here means the whole ray goes through. Total internal reflection is the one
    exception, and not an optional one - past the critical angle no transmitted direction
    exists at all - so it still turns the ray around. That is also what puts the bright rim on
    a previewed glass sphere.

    Color is the material color applied once per interface the ray gets through, not the
    Beer-Lambert absorption of the path tracer: a preview traces no path whose length inside
    the solid it could integrate over. Preview and Path therefore agree on which surfaces are
    see-through and on how they bend light, but not on the depth of the color - a closed solid
    picks up its color twice here, on the way in and on the way out. A ray turned back by
    total internal reflection never crossed the boundary, so it picks up nothing.
*/
DEVICE inline void direct_tracer_transmit(vec3<float>& direction,
                                          RGB<float>& tint,
                                          const Material& m,
                                          const RGB<float>& shading_color,
                                          const vec3<float>& n,
                                          const vec3<float>& v,
                                          bool backfacing)
    {
    // A ray on its way out of the material meets the same interface from the dense side, so
    // the two indices swap.
    const float eta_i = backfacing ? m.ior : 1.0f;
    const float eta_t = backfacing ? 1.0f : m.ior;

    vec3<float> l;
    if (refract(l, v, n, eta_i / eta_t))
        {
        tint *= m.getColor(shading_color);
        }
    else
        {
        // total internal reflection leaves the mirror direction as the only option
        l = 2.0f * dot(n, v) * n - v;
        }

    direction = l;
    }

//! Shade one hit against the lights
/*! \param m Material at the hit
    \param shading_color Color of the primitive
    \param n Normal, already flipped to the side the ray arrived from
    \param v Unit vector pointing back along the incoming ray
    \param lights Lights in scene coordinates

    \returns The linear space color of the surface

    No secondary rays and no shadow rays: each light is reduced to the point on it closest to
    the mirror direction and the material's BRDF is evaluated there.
*/
DEVICE inline RGB<float> direct_tracer_shade(const Material& m,
                                             const RGB<float>& shading_color,
                                             const vec3<float>& n,
                                             const vec3<float>& v,
                                             const Lights& lights)
    {
    if (m.isSolid())
        {
        return m.getColor(shading_color);
        }

    RGB<float> c(0, 0, 0);
    for (unsigned int light_id = 0; light_id < lights.N; light_id++)
        {
        vec3<float> l = lights.direction[light_id];

        // find the representative point, a vector pointing
        // to the a point on the area light with a smallest
        // angle to the reflection vector
        vec3<float> r = -v + (2.0f * n * dot(n, v));

        // find the closest point on the area light
        float half_angle = lights.theta[light_id];
        float cos_half_angle = cosf(half_angle);
        float ldotr = dot(l, r);
        if (ldotr < cos_half_angle)
            {
            vec3<float> a = cross(l, r);
            a = a / sqrtf(dot(a, a));

            // miss the light, need to rotate r by the
            // difference in the angles about l cross r
            quat<float> q = quat<float>::fromAxisAngle(a, -acosf(ldotr) + half_angle);
            r = rotate(q, r);
            }
        else
            {
            // hit the light, no modification necessary to r
            }

        // only apply brdf when the light faces the surface
        RGB<float> f_d;
        float ndotl = dot(n, l);
        if (ndotl >= 0.0f)
            f_d = m.brdf_diffuse(l, v, n, shading_color) * ndotl;
        else
            f_d = RGB<float>(0.0f, 0.0f, 0.0f);

        RGB<float> f_s;
        if (dot(n, r) >= 0.0f)
            {
            f_s = m.brdf_specular(r, v, n, shading_color, half_angle) * dot(n, r);
            }
        else
            f_s = RGB<float>(0.0f, 0.0f, 0.0f);

        c += (f_d + f_s) * float(M_PI) * lights.color[light_id];
        }

    return c;
    }

    } // namespace fresnel
#undef DEVICE

#endif
