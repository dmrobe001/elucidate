// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef TRACER_METHODS_H_
#define TRACER_METHODS_H_

#include "BVH.h"
#include "Scene.h"
#include "common/Camera.h"
#include "common/ColorMath.h"
#include "common/Light.h"
#include "common/RayGen.h"
#include "common/TracerPathMethods.h"

/*! \file TracerMethods.h
    \brief The per-pixel body of each tracer.

    TracerDirect.cu and TracerPath.cu reduce to a thread index and a buffer write around these two
   functions. Keeping the work here rather than inside the __global__ functions mirrors how
   common/TracerPathMethods.h factors the shading kernel out of the CPU render loop, and it lets the
   same code be driven from the host.

    This header reaches common/RayGen.h, which cannot be compiled by a host compiler once
   cuda_runtime.h has been included: Random123's boxmuller.hpp takes CUDART_VERSION to mean that the
   CUDA math library will supply sincospif. Include it from nvcc compiled translation units only -
   the pybind11 glue needs the Tracer classes, not their kernels.
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
//! Shade one pixel against the lights
/*! \param scene The committed scene to trace into
    \param cam Camera to generate rays with
    \param lights Lights in scene coordinates
    \param background_color Color to return when a ray hits nothing
    \param background_alpha Alpha to return when a ray hits nothing
    \param aa_n Number of AA samples in each direction
    \param i Pixel index in the x direction
    \param j Pixel index in the y direction

    \returns The linear space color of the pixel

    This is a direct port of the body of TracerDirect::renderImplementation() in the CPU backend:
   rtcIntersect1() becomes bvh_intersect() and the tiled TBB loop becomes the thread grid of the
   kernel that calls this. It is DEVICE rather than __device__ so that the same code can be driven
   from the host, which is the only way to exercise it without a GPU.
*/
DEVICE inline RGBA<float> trace_direct_pixel(const SceneView& scene,
                                             const Camera& cam,
                                             const Lights& lights,
                                             const RGB<float>& background_color,
                                             float background_alpha,
                                             unsigned int aa_n,
                                             unsigned int i,
                                             unsigned int j)
    {
    // loop over AA samples
    RGBA<float> output_avg(0, 0, 0, 0);

    for (unsigned int sample = 0; sample < aa_n * aa_n; sample++)
        {
        // trace a ray into the scene
        vec3<float> org, dir;
        cam.generateRay(org, dir, i, j, sample);

        const HitInfo hit = bvh_intersect(scene.bvh, scene.geometry, org, dir, 0.0f, HUGE_VALF);

        // determine the output pixel color
        RGB<float> c = background_color;
        float a = background_alpha;

        if (hit.hit())
            {
            vec3<float> n = hit.Ng;
            n /= sqrtf(dot(n, n));
            vec3<float> v = -dir / sqrtf(dot(dir, dir));

            // Geometry reports the geometric normal, which points out of the primitive
            // whichever side the ray struck. Flip it to the side the ray arrived from so that
            // back faces shade instead of going black.
            if (dot(n, v) < 0.0f)
                {
                n = -n;
                }

            Material m;

            // apply the material color or outline color depending on
            // the distance to the edge
            if (hit.d >= scene.outline_widths[hit.geom_id])
                m = scene.materials[hit.geom_id];
            else
                m = scene.outline_materials[hit.geom_id];

            if (m.isSolid())
                {
                c = m.getColor(hit.shading_color);
                }
            else
                {
                c = RGB<float>(0, 0, 0);
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
                        f_d = m.brdf_diffuse(l, v, n, hit.shading_color) * ndotl;
                    else
                        f_d = RGB<float>(0.0f, 0.0f, 0.0f);

                    RGB<float> f_s;
                    if (dot(n, r) >= 0.0f)
                        {
                        f_s = m.brdf_specular(r, v, n, hit.shading_color, half_angle) * dot(n, r);
                        }
                    else
                        f_s = RGB<float>(0.0f, 0.0f, 0.0f);

                    c += (f_d + f_s) * float(M_PI) * lights.color[light_id];
                    }
                }

            a = 1.0;
            }

        // accumulate importance sampled average
        output_avg += RGBA<float>(c, a);
        } // end loop over AA samples

    return output_avg / float(aa_n * aa_n);
    }

//! Trace one path sample for one pixel
/*! \param scene The committed scene to trace into
    \param cam Camera to generate rays with
    \param lights Lights in scene coordinates
    \param background_color Color to return when a ray hits nothing
    \param background_alpha Alpha to return when a ray hits nothing
    \param light_samples Number of paths to trace from the first hit
    \param n_samples Index of this sample, the first is 1
    \param seed Random number seed
    \param width Width of the output image in pixels
    \param height Height of the output image in pixels
    \param i Pixel index in the x direction
    \param j Pixel index in the y direction

    \returns The linear space contribution of this sample, before it is averaged in

    This is a direct port of the body of TracerPath::renderImplementation() in the CPU backend. The
   light sample loop, the depth loop and the hit/miss dispatch are carried over unchanged; the RNG
   is keyed on (pixel, seed, depth, sample) and never on traversal order, so a pixel produces the
   same path here as it does on the CPU. It is DEVICE rather than __device__ so that the same code
   can be driven from the host, which is the only way to exercise it without a GPU.
*/
DEVICE inline RGBA<float> trace_path_sample(const SceneView& scene,
                                            const Camera& cam,
                                            const Lights& lights,
                                            const RGB<float>& background_color,
                                            float background_alpha,
                                            unsigned int light_samples,
                                            unsigned int n_samples,
                                            unsigned int seed,
                                            unsigned int width,
                                            unsigned int height,
                                            unsigned int i,
                                            unsigned int j)
    {
    // create the ray generator for this pixel
    const RayGen ray_gen(i, j, width, height, seed);

    // per ray data
    PRDpath prd;
    prd.result = RGB<float>(0, 0, 0);
    prd.a = 1.0f;

    // trace the first ray into the scene
    vec3<float> org_initial, dir_initial;
    cam.generateRay(org_initial, dir_initial, i, j, n_samples);

    const HitInfo hit_initial
        = bvh_intersect(scene.bvh, scene.geometry, org_initial, dir_initial, 1e-3f, HUGE_VALF);

    // trace a path from the hit point into the scene light_samples times
    for (prd.light_sample = 0; prd.light_sample < light_samples; prd.light_sample++)
        {
        prd.attenuation = RGB<float>(1.0f, 1.0f, 1.0f);
        prd.done = false;
        prd.specular_path = true;

        for (prd.depth = 0;; prd.depth++)
            {
            vec3<float> org, dir;
            HitInfo hit;

            if (prd.depth == 0)
                {
                // the first hit is cached above
                org = org_initial;
                dir = dir_initial;
                hit = hit_initial;
                }
            else
                {
                org = prd.origin;
                dir = prd.direction;
                hit = bvh_intersect(scene.bvh, scene.geometry, org, dir, 1e-3f, HUGE_VALF);
                }

            if (hit.hit())
                {
                // call hit program
                path_tracer_hit(prd,
                                scene.materials[hit.geom_id],
                                scene.outline_materials[hit.geom_id],
                                hit.d,
                                scene.outline_widths[hit.geom_id],
                                hit.shading_color,
                                hit.Ng,
                                org,
                                dir,
                                hit.t,
                                ray_gen,
                                n_samples,
                                light_samples);
                }
            else
                {
                // call miss program
                path_tracer_miss(prd,
                                 background_color,
                                 background_alpha,
                                 light_samples,
                                 lights,
                                 dir);
                }

            // break out of the loop when done
            if (prd.done)
                break;
            } // end depth loop
        } // end light samples loop

    return RGBA<float>(prd.result / float(light_samples), prd.a);
    }

    } // namespace gpu
    } // namespace fresnel

#undef DEVICE

#endif
