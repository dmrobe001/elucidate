// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "TracerMethods.h"
#include "TracerPath.h"
#include <cmath>
#include <stdexcept>

namespace fresnel
    {
namespace gpu
    {
namespace kernel
    {
//! Trace one path sample per thread
__global__ void path(RGBA<float>* linear_output,
                     RGBA<unsigned char>* srgb_output,
                     unsigned int width,
                     unsigned int height,
                     unsigned int row_start,
                     unsigned int row_count,
                     SceneView scene,
                     Camera cam,
                     Lights lights,
                     RGB<float> background_color,
                     float background_alpha,
                     unsigned int light_samples,
                     unsigned int n_samples,
                     unsigned int seed,
                     bool highlight_warning,
                     RGB<float> highlight_warning_color)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || local_j >= row_count)
        return;

    // the pixel coordinate passed on below is the same one an unsplit launch would use: only
    // which launch computes it, not the value itself, depends on the row band
    const unsigned int j = row_start + local_j;

    const RGBA<float> output_sample = trace_path_sample(scene,
                                                        cam,
                                                        lights,
                                                        background_color,
                                                        background_alpha,
                                                        light_samples,
                                                        n_samples,
                                                        seed,
                                                        width,
                                                        height,
                                                        i,
                                                        j);

    // running average using Welford's method. Variance is not particularly useful to users, so
    // don't compute that.
    // (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
    const unsigned int pixel = j * width + i;
    const RGBA<float> old_mean = linear_output[pixel];
    linear_output[pixel] = old_mean + (output_sample - old_mean) / float(n_samples);

    // convert the current average output to sRGB
    const RGBA<float> output_pixel = linear_output[pixel];
    if (!highlight_warning
        || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
        srgb_output[pixel] = sRGB(output_pixel);
    else
        srgb_output[pixel] = sRGB(RGBA<float>(highlight_warning_color, output_pixel.a));
    }

    } // namespace kernel

/*! \param device Device to attach the raytracer to
 */
TracerPath::TracerPath(std::shared_ptr<Device> device,
                       unsigned int w,
                       unsigned int h,
                       unsigned int light_samples)
    : Tracer(device, w, h), m_light_samples(light_samples)
    {
    reset();
    }

TracerPath::~TracerPath() { }

void TracerPath::reset()
    {
    m_n_samples = 0;
    m_seed++;

    m_device->makeCurrent();

    RGBA<float>* linear_output = m_linear_out->map();
    memset((void*)linear_output,
           0,
           sizeof(RGBA<float>) * m_linear_out->getW() * m_linear_out->getH());
    m_linear_out->unmap();

    RGBA<unsigned char>* srgb_output = m_srgb_out->map();
    memset((void*)srgb_output,
           0,
           sizeof(RGBA<unsigned char>) * m_linear_out->getW() * m_linear_out->getH());
    m_srgb_out->unmap();
    }

void TracerPath::render(std::shared_ptr<Scene> scene)
    {
    Tracer::render(scene);
    renderImplementation(scene);
    }

void TracerPath::renderImplementation(std::shared_ptr<Scene> scene)
    {
    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();

    const Camera cam(scene->getCamera(), m_linear_out->getW(), m_linear_out->getH(), m_seed);
    const Lights lights(scene->getLights(), cam);

    // update the acceleration structure and the device side materials
    scene->commit();

    RGBA<float>* linear_output = m_linear_out->map();
    RGBA<unsigned char>* srgb_output = m_srgb_out->map();

    // update number of samples (the first sample is 1)
    m_n_samples++;

    const unsigned int height = m_linear_out->getH();
    const unsigned int width = m_linear_out->getW();

    const dim3 block(trace_block_dim, trace_block_dim);

    m_chunker.run(width,
                  height,
                  [&](unsigned int row_start, unsigned int row_count)
                  {
                      const dim3 grid((width + block.x - 1) / block.x,
                                      (row_count + block.y - 1) / block.y);

                      kernel::path<<<grid, block>>>(linear_output,
                                                    srgb_output,
                                                    width,
                                                    height,
                                                    row_start,
                                                    row_count,
                                                    scene->getSceneView(),
                                                    cam,
                                                    lights,
                                                    background_color,
                                                    background_alpha,
                                                    m_light_samples,
                                                    m_n_samples,
                                                    m_seed,
                                                    m_highlight_warning,
                                                    m_highlight_warning_color);
                      CUDA_CHECK_LAUNCH();
                  });

    m_linear_out->unmap();
    m_srgb_out->unmap();
    }

/*! \param m Python module to export in
 */
void export_TracerPath(pybind11::module& m)
    {
    pybind11::class_<TracerPath, Tracer, std::shared_ptr<TracerPath>>(m, "TracerPath")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int, unsigned int>())
        .def("getNumSamples", &TracerPath::getNumSamples)
        .def("reset", &TracerPath::reset)
        .def("setLightSamples", &TracerPath::setLightSamples);
    }

    } // namespace gpu
    } // namespace fresnel
