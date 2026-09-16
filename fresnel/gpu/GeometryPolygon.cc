// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryPolygon.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the polygon (in counterclockwise order)
    \param rounding_radius The rounding radius of the spheropolygon
    \param N number of primitives

    Initialize the polygon geometry.
*/
GeometryPolygon::GeometryPolygon(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
    float rounding_radius,
    unsigned int N)
    : Geometry(scene), m_rounding_radius(rounding_radius)
    {
    m_device->makeCurrent();

    // allocate buffer data
    m_position = std::shared_ptr<Array<vec2<float>>>(new Array<vec2<float>>(N));
    m_angle = std::shared_ptr<Array<float>>(new Array<float>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // copy the vertices from the numpy array to internal storage
    pybind11::buffer_info info = vertices.request();

    if (info.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info.shape[1] != 2)
        throw std::runtime_error("vertices must be a Nvert by 2 array");

    float* verts_f = (float*)info.ptr;

    m_vertices.resize(info.shape[0]);

    for (unsigned int i = 0; i < info.shape[0]; i++)
        {
        vec2<float> p0(verts_f[i * 2], verts_f[i * 2 + 1]);

        m_vertices[i] = p0;

        // precompute radius in the xy plane
        m_radius = std::max(m_radius, sqrtf(dot(p0, p0)));
        }
    // pad the radius with the rounding radius
    m_radius += m_rounding_radius;

    // describe the geometry for the traversal
    m_data.type = geometry_polygon;
    m_data.n_primitives = N;
    m_data.position_2d = m_position->getPointer();
    m_data.angle = m_angle->getPointer();
    m_data.color = m_color->getPointer();
    m_data.polygon_vertices = m_vertices.get();
    m_data.n_polygon_vertices = (unsigned int)m_vertices.size();
    m_data.rounding_radius = m_rounding_radius;
    m_data.polygon_radius = m_radius;

    attach();

    // set default material
    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));
    }

GeometryPolygon::~GeometryPolygon() { }

/*! \param m Python module to export in
 */
void export_GeometryPolygon(pybind11::module& m)
    {
    pybind11::class_<GeometryPolygon, Geometry, std::shared_ptr<GeometryPolygon>>(m,
                                                                                  "GeometryPolygon")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             float,
             unsigned int>())
        .def("getPositionBuffer", &GeometryPolygon::getPositionBuffer)
        .def("getAngleBuffer", &GeometryPolygon::getAngleBuffer)
        .def("getColorBuffer", &GeometryPolygon::getColorBuffer)
        .def("getRadius", &GeometryPolygon::getRadius);
    }

    } // namespace gpu
    } // namespace fresnel
