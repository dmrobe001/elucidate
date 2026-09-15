// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "ColorMath.h"
#include "Light.h"
#include "VectorMath.h"

#ifndef __TRACER_PATH_METHODS_H__
#define __TRACER_PATH_METHODS_H__

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
//! Per ray data for path tracer rays
struct PRDpath
    {
    RGB<float> result;
    float a;
    RGB<float> attenuation;
    vec3<float> origin;
    vec3<float> direction;
    unsigned int light_sample;
    unsigned int depth;
    bool done;

    //! True while every scatter so far has been a delta event
    /*! The background is a backdrop rather than an environment light: it lights nothing, and a
        diffuse or glossy bounce loses sight of it. A path that has only reflected off or
        refracted through smooth interfaces is still looking straight out at it, though, which
        is what lets the backdrop show through glass and in mirrors.
    */
    bool specular_path;
    };

DEVICE void path_tracer_miss(PRDpath& prd,
                             const RGB<float>& _background_color,
                             const float _background_alpha,
                             unsigned int _light_samples,
                             const Lights& _lights,
                             const vec3<float>& ray_direction)

    {
    if (prd.depth == 0)
        {
        // a miss at depth 0 implies we hit the background on the primary ray
        // no need to resample the background N times
        prd.result = _background_color * float(_light_samples);
        prd.a = _background_alpha;
        }
    else
        {
        // on subsequent rays, process the area lights
        // see if it hit any lights and compute the output color accordingly
        for (unsigned int light_id = 0; light_id < _lights.N; light_id++)
            {
            vec3<float> l = _lights.direction[light_id];
            float half_angle = _lights.theta[light_id];
            float cos_half_angle = cosf(half_angle);
            if (half_angle > float(M_PI) / 2.0f)
                half_angle = float(M_PI) / 2.0f;
            float ldotd = dot(l, vec3<float>(ray_direction));

            // NB: we can do this because ray directions are always normalized
            if (ldotd >= cos_half_angle)
                {
                // hit the light
                // the division bin sin(light half angle) normalizes the lights so that
                // a light of color 1 of any non-zero size results in an output of 1
                // when that light is straight over the surface
                prd.result += prd.attenuation * _lights.color[light_id] / sinf(half_angle);
                }
            } // end loop over lights

        // the backdrop is still in view along a path that has only passed through smooth
        // interfaces, so it shows through glass and in mirrors
        if (prd.specular_path)
            {
            prd.result += prd.attenuation * _background_color;
            }
        }

    prd.done = true;
    }

DEVICE void path_tracer_hit(PRDpath& prd,
                            const Material& _material,
                            const Material& _outline_material,
                            const float _shading_distance,
                            const float _outline_width,
                            const RGB<float>& _shading_color,
                            const vec3<float>& _shading_normal,
                            const vec3<float>& ray_origin,
                            const vec3<float>& ray_direction,
                            const float _t_hit,
                            const RayGen& ray_gen,
                            const unsigned int _n_samples,
                            const unsigned int _light_samples)
    {
    Material m;

    // apply the material color or outline color depending on the distance to the edge
    if (_shading_distance >= _outline_width)
        {
        m = _material;
        }
    else
        {
        m = _outline_material;
        }

    // Geometry reports the *geometric* normal, which points out of the primitive whichever side
    // the ray struck. Flip it to the side the ray arrived from before shading, and remember
    // which face was hit: a back face is the far wall of a solid seen from the inside, which is
    // exactly where a transmitted path continues.
    vec3<float> n = _shading_normal * fast::rsqrt(dot(_shading_normal, _shading_normal));
    vec3<float> v = -vec3<float>(ray_direction);

    const bool backfacing = (dot(n, v) < 0.0f);
    if (backfacing)
        {
        n = -n;
        }

    if (m.isSolid())
        {
        // testing: treat solid colors as emitters
        // when the ray hits an emitter, terminate the path and evaluate the final color
        prd.result += prd.attenuation * m.getColor(_shading_color);
        prd.done = true;
        }
    else
        {
        // when the ray hits an object with a normal material, update the attenuation using the BRDF
        // choose a random direction l to continue the path.
        float factor = 1.0;
        ScatterEvent event = scatter_diffuse_or_glossy;

        vec3<float> l
            = ray_gen.sampleScatterDirection(factor,
                                             event,
                                             v,
                                             n,
                                             backfacing,
                                             prd.depth,
                                             (_n_samples - 1) * _light_samples + prd.light_sample,
                                             m);

        if (event == scatter_specular_transmission)
            {
            // Tint the transmitted ray. Entry and exit each apply the square root so that a
            // full crossing multiplies to the material color.
            // TODO: replace with distance dependent Beer-Lambert absorption, which is what
            // makes thick parts of a solid read as deeper in color than thin ones.
            RGB<float> trans_color = m.getColor(_shading_color);
            trans_color.r = sqrtf(trans_color.r);
            trans_color.g = sqrtf(trans_color.g);
            trans_color.b = sqrtf(trans_color.b);
            prd.attenuation *= trans_color * factor;
            }
        else if (event == scatter_specular_reflection)
            {
            // Reflection off a dielectric interface is not tinted by the body color: it is
            // light that never entered the material.
            prd.attenuation *= factor;
            }
        else
            {
            // a diffuse or glossy bounce scatters over a lobe rather than a direction, so the
            // path is no longer looking straight out at the backdrop
            prd.specular_path = false;

            float ndotl = dot(n, l);

            // n faces the viewer by construction, so only the sampled light direction can end
            // up behind the surface. Back faces are shaded rather than dropped: terminating
            // them here would black out every path that continues inside a solid.
            if (ndotl > 0.0f)
                {
                prd.attenuation *= m.brdf(l, v, n, _shading_color) * ndotl * factor;
                }
            else
                {
                prd.attenuation = RGB<float>(0, 0, 0);
                prd.done = true;
                }
            }

        // set the origin and direction for the next ray in the path
        prd.origin = ray_origin + ray_direction * _t_hit;
        prd.direction = l;

        // break out of the loop when attenuation is small (Russian roulette termination )
        if (ray_gen.shouldTerminatePath(prd.attenuation,
                                        prd.depth,
                                        (_n_samples - 1) * _light_samples + prd.light_sample))
            {
            prd.done = true;
            }
        }
    }
    } // namespace fresnel
#undef DEVICE

#endif
