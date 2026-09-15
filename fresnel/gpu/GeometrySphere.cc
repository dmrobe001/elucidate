// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometrySphere.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of spheres to manage

    Initialize the sphere geometry.
*/
GeometrySphere::GeometrySphere(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
    {
    m_device->makeCurrent();

    // initialize the buffers
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
    m_radius = std::shared_ptr<Array<float>>(new Array<float>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // describe the geometry for the traversal
    m_data.type = geometry_sphere;
    m_data.n_primitives = N;
    m_data.position = m_position->getPointer();
    m_data.radius = m_radius->getPointer();
    m_data.color = m_color->getPointer();

    attach();

    // set default material
    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));
    }

GeometrySphere::~GeometrySphere() { }

/*! \param m Python module to export in
 */
void export_GeometrySphere(pybind11::module& m)
    {
    pybind11::class_<GeometrySphere, Geometry, std::shared_ptr<GeometrySphere>>(m, "GeometrySphere")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPositionBuffer", &GeometrySphere::getPositionBuffer)
        .def("getRadiusBuffer", &GeometrySphere::getRadiusBuffer)
        .def("getColorBuffer", &GeometrySphere::getColorBuffer);
    }

    } // namespace gpu
    } // namespace fresnel
