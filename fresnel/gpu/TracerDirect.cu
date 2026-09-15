// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "TracerDirect.h"
#include "TracerMethods.h"
#include <cmath>
#include <stdexcept>

namespace fresnel
    {
namespace gpu
    {
namespace kernel
    {
//! Shade one pixel per thread
__global__ void direct(RGBA<float>* linear_output,
                       RGBA<unsigned char>* srgb_output,
                       unsigned int width,
                       unsigned int height,
                       SceneView scene,
                       Camera cam,
                       Lights lights,
                       RGB<float> background_color,
                       float background_alpha,
                       unsigned int aa_n,
                       bool highlight_warning,
                       RGB<float> highlight_warning_color)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height)
        return;

    const RGBA<float> output_pixel
        = trace_direct_pixel(scene, cam, lights, background_color, background_alpha, aa_n, i, j);

    // write the output pixel
    const unsigned int pixel = j * width + i;

    linear_output[pixel] = output_pixel;
    if (!highlight_warning
        || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
        srgb_output[pixel] = sRGB(output_pixel);
    else
        srgb_output[pixel] = sRGB(RGBA<float>(highlight_warning_color, output_pixel.a));
    }

    } // namespace kernel

/*! \param device Device to attach the raytracer to
 */
TracerDirect::TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    }

TracerDirect::~TracerDirect() { }

void TracerDirect::render(std::shared_ptr<Scene> scene)
    {
    Tracer::render(scene);
    renderImplementation(scene);
    }

void TracerDirect::renderImplementation(std::shared_ptr<Scene> scene)
    {
    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();

    // direct tracers do not support depth of field
    UserCamera user_camera = scene->getCamera();
    user_camera.f_stop = std::numeric_limits<float>::infinity();

    // disable aa sampling with m_aa_n is 1
    bool sample_aa = (m_aa_n != 1);

    const Camera cam(user_camera, m_linear_out->getW(), m_linear_out->getH(), m_seed, sample_aa);
    const Lights lights(scene->getLights(), cam);

    // update the acceleration structure and the device side materials
    scene->commit();

    RGBA<float>* linear_output = m_linear_out->map();
    RGBA<unsigned char>* srgb_output = m_srgb_out->map();

    const unsigned int height = m_linear_out->getH();
    const unsigned int width = m_linear_out->getW();

    const dim3 block(trace_block_dim, trace_block_dim);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    kernel::direct<<<grid, block>>>(linear_output,
                                    srgb_output,
                                    width,
                                    height,
                                    scene->getSceneView(),
                                    cam,
                                    lights,
                                    background_color,
                                    background_alpha,
                                    m_aa_n,
                                    m_highlight_warning,
                                    m_highlight_warning_color);
    CUDA_CHECK_LAUNCH();

    m_linear_out->unmap();
    m_srgb_out->unmap();
    }

/*! \param m Python module to export in
 */
void export_TracerDirect(pybind11::module& m)
    {
    pybind11::class_<TracerDirect, Tracer, std::shared_ptr<TracerDirect>>(m, "TracerDirect")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("setAntialiasingN", &TracerDirect::setAntialiasingN)
        .def("getAntialiasingN", &TracerDirect::getAntialiasingN);
    }

    } // namespace gpu
    } // namespace fresnel
