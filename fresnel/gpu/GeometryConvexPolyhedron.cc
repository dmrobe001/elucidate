// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryConvexPolyhedron.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param plane_origins Origins of the planes that make up the polyhedron
    \param plane_normals Normals of the planes that make up the polyhedron
    \param plane_colors Colors of the planes that make up the polyhedron
    \param N number of primitives
    \param r radius of the polyhedron

    Initialize the polyhedron geometry.
*/
GeometryConvexPolyhedron::GeometryConvexPolyhedron(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_origins,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_normals,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_colors,
    unsigned int N,
    float r)
    : Geometry(scene)
    {
    m_device->makeCurrent();

    // allocate buffer data
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
    m_orientation = std::shared_ptr<Array<quat<float>>>(new Array<quat<float>>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // access the plane data
    pybind11::buffer_info info_origin = plane_origins.request();

    if (info_origin.ndim != 2)
        throw std::runtime_error("plane_origins must be a 2-dimensional array");

    if (info_origin.shape[1] != 3)
        throw std::runtime_error("plane_origins must be a Nvert by 3 array");

    float* origin_f = (float*)info_origin.ptr;

    pybind11::buffer_info info_normal = plane_normals.request();

    if (info_normal.ndim != 2)
        throw std::runtime_error("plane_normals must be a 2-dimensional array");

    if (info_normal.shape[1] != 3)
        throw std::runtime_error("plane_normals must be a Nvert by 3 array");

    if (info_normal.shape[0] != info_origin.shape[0])
        throw std::runtime_error("Number of vertices must match in origin and normal arrays");

    float* normal_f = (float*)info_normal.ptr;

    pybind11::buffer_info info_color = plane_colors.request();

    if (info_color.ndim != 2)
        throw std::runtime_error("plane_colors must be a 2-dimensional array");

    if (info_color.shape[1] != 3)
        throw std::runtime_error("plane_colors must be a Nvert by 3 array");

    if (info_color.shape[0] != info_origin.shape[0])
        throw std::runtime_error("Number of vertices must match in origin and color arrays");

    float* color_f = (float*)info_color.ptr;

    // construct planes in device visible storage
    const unsigned int n_planes = (unsigned int)info_normal.shape[0];
    m_plane_origin.resize(n_planes);
    m_plane_normal.resize(n_planes);
    m_plane_color.resize(n_planes);

    for (unsigned int i = 0; i < n_planes; i++)
        {
        vec3<float> n(normal_f[i * 3], normal_f[i * 3 + 1], normal_f[i * 3 + 2]);
        n = n / sqrtf(dot(n, n));

        m_plane_origin[i] = vec3<float>(origin_f[i * 3], origin_f[i * 3 + 1], origin_f[i * 3 + 2]);
        m_plane_normal[i] = vec3<float>(n.x, n.y, n.z);
        m_plane_color[i] = RGB<float>(color_f[i * 3], color_f[i * 3 + 1], color_f[i * 3 + 2]);
        }

    // for now, take a user supplied radius
    m_radius = r;

    // describe the geometry for the traversal
    m_data.type = geometry_convex_polyhedron;
    m_data.n_primitives = N;
    m_data.position = m_position->getPointer();
    m_data.orientation = m_orientation->getPointer();
    m_data.color = m_color->getPointer();
    m_data.plane_origin = m_plane_origin.get();
    m_data.plane_normal = m_plane_normal.get();
    m_data.plane_color = m_plane_color.get();
    m_data.n_planes = n_planes;
    m_data.polyhedron_radius = m_radius;
    m_data.color_by_face = m_color_by_face;

    attach();

    // set default material
    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));
    }

GeometryConvexPolyhedron::~GeometryConvexPolyhedron() { }

/*! \param m Python module to export in
 */
void export_GeometryConvexPolyhedron(pybind11::module& m)
    {
    pybind11::class_<GeometryConvexPolyhedron, Geometry, std::shared_ptr<GeometryConvexPolyhedron>>(
        m,
        "GeometryConvexPolyhedron")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int,
             float>())
        .def("getPositionBuffer", &GeometryConvexPolyhedron::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryConvexPolyhedron::getOrientationBuffer)
        .def("getColorBuffer", &GeometryConvexPolyhedron::getColorBuffer)
        .def("setColorByFace", &GeometryConvexPolyhedron::setColorByFace)
        .def("getColorByFace", &GeometryConvexPolyhedron::getColorByFace);
    }

    } // namespace gpu
    } // namespace fresnel
