// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef TRACER_H_
#define TRACER_H_

#include "cuda_platform.h"
#include <pybind11/pybind11.h>

#include "Array.h"
#include "Scene.h"
#include "common/Camera.h"
#include "common/ColorMath.h"

#include <algorithm>
#include <chrono>

namespace fresnel
    {
namespace gpu
    {
//! Split a per-pixel kernel into row bands sized to stay under a target launch duration
/*! A launch that runs too long on a display-driving (WDDM) GPU trips the driver's hang-detection
    timeout, which resets the device and takes every process using it down with it. Neither the
    image resolution nor the cost of any given pixel bounds a launch's duration on its own - a
    complex scene at a middling resolution can be exactly as slow as a simple one at a large
    resolution - so this measures each launch and adjusts the next one's size to hold a target
    duration, rather than picking a fixed tile size.

    Tracking a pixel budget rather than a row count means a resize is handled for free: the row
    count a call asks for is recomputed from the budget and the new width every time, so a window
    resize does not by itself invalidate what was learned about this GPU's throughput.

    Splitting an image into row bands changes nothing about what any pixel computes: each launch
    is passed the same width, height, and seed as an unsplit launch would have been, only a
    different row_start/row_count, and every DEVICE routine keys its RNG draws on the pixel
    coordinate, never on how the image happened to be divided up to compute it.
*/
class AdaptiveChunker
    {
    public:
    //! Render an image in row bands, timing and resizing them to hold a target duration
    /*! \param width Image width, in pixels
        \param height Image height, in pixels
        \param launch_rows Called as launch_rows(row_start, row_count) to render one band and
               synchronize with it before returning, so the elapsed time reflects the device work
    */
    template<class F> void run(unsigned int width, unsigned int height, F launch_rows)
        {
        unsigned int row = 0;
        while (row < height)
            {
            const unsigned int rows
                = std::min(height - row, std::max(1u, m_pixel_budget / std::max(1u, width)));

            const auto t0 = std::chrono::steady_clock::now();
            launch_rows(row, rows);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // hold the launch duration near target_ms: halve the budget on an overshoot so a
            // single slow band cannot repeat, double it on a wide margin so a fast GPU is not
            // held to an unnecessarily conservative tile size forever
            if (ms > target_ms)
                m_pixel_budget = std::max(min_pixel_budget, m_pixel_budget / 2);
            else if (ms < target_ms * 0.25)
                m_pixel_budget = std::min(max_pixel_budget, m_pixel_budget * 2);

            row += rows;
            }
        }

    private:
    static constexpr double target_ms = 200.0; //!< Duration a launch should stay under
    static constexpr unsigned int min_pixel_budget = 1024; //!< Never split finer than this
    static constexpr unsigned int max_pixel_budget = 1u << 22; //!< Stop growing past this

    //! Current estimate of how many pixels this GPU can shade within target_ms
    /*! Deliberately conservative: the first launch after startup or a scene/camera change is a
        guess with no data behind it yet, and the failure mode for guessing too high is a hung
        GPU, so a wrong guess only ever costs a few extra launches of latency, never a crash.
    */
    unsigned int m_pixel_budget = min_pixel_budget;
    };

//! Base class for the raytracer
/*! The base class Tracer specifies common methods used for all tracers. This includes output buffer
   management, and defining the rendering API.

    The output buffer is stored in two formats. *m_linear_out* store the ray traced output in a
   linear RGB color space. This buffer is suitable for tone mapping and averaging with other render
   output. *m_srgb_out* stores the output in the sRGB color space and in a 4 bytes per pixel format
   suitable for direct use in image display.

    The rendering methods themselves do nothing. Derived classes must implement them.
*/
class Tracer
    {
    public:
    //! Constructor
    Tracer(std::shared_ptr<Device> device, unsigned int w, unsigned int h);

    //! Destructor
    virtual ~Tracer();

    //! Resize the output buffer
    virtual void resize(unsigned int w, unsigned int h);

    //! Render a scene
    virtual void render(std::shared_ptr<Scene> scene);

    //! Get the SRGB output pixel buffer
    virtual std::shared_ptr<Array<RGBA<unsigned char>>> getSRGBOutputBuffer()
        {
        return m_srgb_out;
        }

    //! Get the SRGB output pixel buffer
    virtual std::shared_ptr<Array<RGBA<float>>> getLinearOutputBuffer()
        {
        return m_linear_out;
        }

    //! Enable highlight warnings
    void enableHighlightWarning(const RGB<float>& color)
        {
        m_highlight_warning = true;
        m_highlight_warning_color = color;
        }

    void disableHighlightWarning()
        {
        m_highlight_warning = false;
        }

    //! Set the random number seed
    void setSeed(unsigned int seed)
        {
        m_seed = seed;
        }

    //! Get the random number seed
    unsigned int getSeed() const
        {
        return m_seed;
        }

    protected:
    std::shared_ptr<Device> m_device; //!< The device the Scene is attached to
    std::shared_ptr<Array<RGBA<float>>> m_linear_out; //!< The output buffer (linear space)
    std::shared_ptr<Array<RGBA<unsigned char>>> m_srgb_out; //!< The output buffer (srgb space)
    bool m_highlight_warning; //!< Set to true to enable highlight warnings in sRGB output
    RGB<float> m_highlight_warning_color; //!< The highlight warning color
    unsigned int m_seed = 0; //!< Random number seed
    AdaptiveChunker m_chunker; //!< Splits each render() into launches sized for this GPU
    };

//! Number of threads in a tracer kernel block
/*! One thread shades one pixel. 8x8 keeps neighbouring pixels in the same warp, which is what makes
   traversal of a coherent camera ray bundle diverge as little as it can in a megakernel.
*/
static const unsigned int trace_block_dim = 8;

//! Export Tracer to python
void export_Tracer(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
