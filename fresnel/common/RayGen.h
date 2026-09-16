// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "Random123/philox.h"
#include "boxmuller.hpp"
#include "uniform.hpp"

#include "ColorMath.h"
#include "Material.h"
#include "VectorMath.h"

#ifndef __RAYGEN_H__
#define __RAYGEN_H__

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
// The following constants go in the 4th counter input to philox
// Each unique use of RNGs must use a different values to get different numbers.

//! Counter for ray path tracing samples (uniform sampling)
const unsigned int rng_val_uniform = 0x11ffabcd;

//! Counter for ray path tracing samples (multiple importance sampling)
const unsigned int rng_val_mis = 0x8754abcd;

//! Counter for ray termination (Russian roulette)
const unsigned int rng_val_rr = 0x54abf853;

//! Hard cap on path depth
/*! Russian roulette alone does not bound a path's length: a lossless dielectric (spec_trans close
    to 1, little or no absorption) keeps the attenuation near 1, so p_continue stays near 1 and a
    path can keep bouncing - most persistently via repeated total internal reflection inside a
    rough transmissive surface at grazing angles - for a very large number of iterations before
    roulette happens to kill it. On the CPU that only slows the one thread computing that pixel.
    On the GPU, a single such path stalls the entire kernel launch until it finishes, since a
    launch cannot return before its slowest thread does; on a display-driving (WDDM) GPU that can
    exceed the driver's hang-detection timeout and take down the whole session. This cap is high
    enough that Russian roulette terminates the overwhelming majority of paths well before it is
    reached, so it is not expected to be visible in any rendered image.
*/
const unsigned int max_path_depth = 64;

//! Ray generation methods
/*! Common code to generate rays on the host and device.
 */
class RayGen
    {
    public:
    //! Default constructor gives uninitialized generator
    DEVICE RayGen() { }

    //! Set ray gen parameters
    DEVICE explicit RayGen(unsigned int i,
                           unsigned int j,
                           unsigned int width,
                           unsigned int height,
                           unsigned int seed)
        : m_width(width), m_height(height), m_i(i), m_j(j)
        {
        unsigned int pixel = j * width + i;

        // create the philox unique key for this RNG which includes the pixel ID and the random seed
        r123::Philox4x32::ukey_type rng_uk = {{pixel, seed}};
        m_rng_key = rng_uk;
        }

    //! Uniform sampling of reflected rays
    /*! \returns The direction to sample next
        \param factor [output] Weighting factor for the sample
        \param v Vector pointing back toward the viewing direction
        \param n Normal vector
        \param depth Depth of the ray in the trace
        \param sample Sample index
    */
    DEVICE vec3<float> uniformSampleReflection(float& factor,
                                               const vec3<float>& v,
                                               const vec3<float>& n,
                                               unsigned int depth,
                                               unsigned int sample) const
        {
        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type rng_counter = {{0, depth, sample, rng_val_uniform}};
        r123::Philox4x32::ctr_type rng_u = rng(rng_counter, m_rng_key);

        // randomly pick a point on the sphere
        r123::float2 rng_gauss1 = r123::boxmuller(rng_u.v[0], rng_u.v[1]);
        r123::float2 rng_gauss2 = r123::boxmuller(rng_u.v[2], rng_u.v[3]);
        vec3<float> l(rng_gauss1.x, rng_gauss1.y, rng_gauss2.x);

        l = l * fast::rsqrt(dot(l, l));

        float ndotl = dot(n, l);
        // l is generated on the whole sphere, if it points down into the surface, make it point up
        if (ndotl < 0.0f)
            {
            l = -l;
            ndotl = -ndotl;
            }
        float pdf = 1.0f / (2.0f * float(M_PI));
        factor = 1.0f / pdf;
        return l;
        }

    //! Sample a scattered direction from a material
    /*! \returns The direction to sample next
        \param factor [output] Weighting factor for the sample
        \param event [output] Which lobe the direction was drawn from
        \param v Vector pointing back toward the viewing direction
        \param n Normal vector, on the same side as \a v
        \param backfacing True when the ray struck the inside of the surface
        \param depth Depth of the ray in the trace
        \param sample Sample index
        \param m Material

        The material is a mixture of an opaque lobe and a smooth dielectric interface, selected
        with probability \a spec_trans. Because each lobe is chosen with the same probability
        as its weight in the mixture, the selection probability cancels and \a factor carries
        only the lobe's own weight.

        The dielectric lobe splits again into reflection and refraction, chosen with the Fresnel
        reflectance so that both branches carry unit weight. \a factor then holds only the
        radiance scaling across the interface.
    */
    DEVICE vec3<float> sampleScatterDirection(float& factor,
                                              ScatterEvent& event,
                                              const vec3<float>& v,
                                              const vec3<float>& n,
                                              bool backfacing,
                                              unsigned int depth,
                                              unsigned int sample,
                                              const Material& m) const
        {
        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type rng_counter = {{0, depth, sample, rng_val_mis}};
        r123::Philox4x32::ctr_type rng_u = rng(rng_counter, m_rng_key);

        // multiple importance sampling
        vec2<float> xi(r123::u01<float>(rng_u.v[0]), r123::u01<float>(rng_u.v[1]));
        float choice_mis = r123::u01<float>(rng_u.v[2]);
        float choice_trans = r123::u01<float>(rng_u.v[3]);

        vec3<float> l;
        if (choice_trans <= m.spec_trans)
            {
            // A ray on its way out of the material meets the same interface from the dense
            // side, so the two indices swap.
            const float eta_i = backfacing ? m.ior : 1.0f;
            const float eta_t = backfacing ? 1.0f : m.ior;
            const float eta = eta_i / eta_t;

            // Matched indices are not a boundary at all, and a boundary that does not exist
            // cannot be rough. Without this the microsurface would shadow light crossing
            // between two identical media.
            if (eta_i == eta_t)
                {
                event = scatter_specular_transmission;
                factor = 1.0f;
                return -v;
                }

            // Scatter about a microfacet normal rather than the surface normal. At roughness
            // 0 the only visible facet is the surface itself and this is an exact mirror or
            // an exact refraction; as roughness grows the interface frosts over.
            const vec3<float> h = m.sampleVisibleNormalGGX(xi, v, n);

            const float F = fresnel_dielectric(dot(h, v), eta_i, eta_t);

            // reflect or refract with the Fresnel reflectance, so that the choice cancels
            // and each branch is left carrying only the visibility of the facet it found
            bool reflect = (choice_mis < F);
            if (!reflect)
                {
                // total internal reflection leaves the mirror direction as the only option
                reflect = !refract(l, v, h, eta);
                }

            if (reflect)
                {
                event = scatter_specular_reflection;
                l = 2.0f * dot(h, v) * h - v;

                // a facet can reflect into the surface it belongs to; nothing escapes there
                factor = (dot(n, l) > 0.0f) ? m.smithG1(dot(n, l)) : 0.0f;
                }
            else
                {
                event = scatter_specular_transmission;

                // likewise a facet can refract back out of the side the ray came from
                factor = (dot(n, l) < 0.0f) ? m.smithG1(dot(n, l)) : 0.0f;

                // radiance is compressed on the way into a denser medium and expanded again
                // on the way out, so a full crossing multiplies back to 1
                factor *= eta * eta;
                }
            }
        else
            {
            // handle reflection with multiple importance sampling
            event = scatter_diffuse_or_glossy;
            if (choice_mis <= 0.5f)
                {
                // diffuse sampling
                l = m.importanceSampleDiffuse(xi, v, n);
                float pdf_diffuse = m.pdfDiffuse(l, v, n);
                float pdf_ggx = m.pdfGGX(l, v, n);
                float w_diffuse = pdf_diffuse / (pdf_diffuse + pdf_ggx);
                factor = w_diffuse / (0.5f * pdf_diffuse);
                }
            else
                {
                // specular reflection
                l = m.importanceSampleGGX(xi, v, n);
                float pdf_diffuse = m.pdfDiffuse(l, v, n);
                float pdf_ggx = m.pdfGGX(l, v, n);
                float w_ggx = pdf_ggx / (pdf_diffuse + pdf_ggx);
                factor = w_ggx / (0.5f * pdf_ggx);
                }
            }
        return l;
        }

    //! Test for Russian roulette ray termination
    /*! \returns True when the path should terminate
        \param attenuation [input/output] Current attenuation
        \param v Vector pointing back toward the viewing direction
        \param depth Depth of the ray in the trace
        \param sample Sample index

         When the path is not terminated, the attenuation is amplified by the appropriate amount.
    */
    DEVICE bool
    shouldTerminatePath(RGB<float>& attenuation, unsigned int depth, unsigned int sample) const
        {
        if (depth >= max_path_depth)
            return true;

        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type rng_counter = {{0, depth, sample, rng_val_rr}};
        r123::Philox4x32::ctr_type rng_u = rng(rng_counter, m_rng_key);

        float p_continue = fmaxf(attenuation.r, fmaxf(attenuation.g, attenuation.b));
        p_continue = fminf(p_continue, 1.0f);
        if (r123::u01<float>(rng_u.v[0]) > p_continue)
            {
            return true;
            }
        else
            {
            attenuation /= p_continue;
            return false;
            }
        }

    protected:
    unsigned int m_width; //!< Width of the output image (in pixels)
    unsigned int m_height; //!< Height of the output image (in pixels)
    r123::Philox4x32::key_type m_rng_key; //!< Key for the random number generator
    unsigned int m_i; //!< i coordinate of the pixel
    unsigned int m_j; //!< j coordinate of the pixel
    };

    } // namespace fresnel
#undef DEVICE

#endif
