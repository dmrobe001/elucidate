// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryMesh.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the mesh
    \param N Number of polyhedra
    Initialize the mesh.
*/
GeometryMesh::GeometryMesh(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
    unsigned int N)
    : Geometry(scene), m_N(N)
    {
    m_device->makeCurrent();

    // extract vertices array from numpy
    pybind11::buffer_info info_vertices = vertices.request();

    if (info_vertices.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a Nvert by 3 array");

    if (info_vertices.shape[0] % 3 != 0)
        throw std::runtime_error("the number of triangle vertices must be a multiple of three.");

    unsigned int n_faces = info_vertices.shape[0] / 3;
    unsigned int n_verts = info_vertices.shape[0];
    float* verts_f = (float*)info_vertices.ptr;

    // allocate buffer data
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
    m_orientation = std::shared_ptr<Array<quat<float>>>(new Array<quat<float>>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(n_verts));

    // copy vertices into local buffer
    m_vertices.resize(n_verts);
    memcpy((void*)m_vertices.get(), verts_f, sizeof(vec3<float>) * n_verts);

    // describe the geometry for the traversal
    m_data.type = geometry_mesh;
    m_data.n_primitives = N * n_faces;
    m_data.position = m_position->getPointer();
    m_data.orientation = m_orientation->getPointer();
    m_data.color = m_color->getPointer();
    m_data.mesh_vertices = m_vertices.get();
    m_data.n_faces = n_faces;

    attach();

    // set default material
    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));
    }

GeometryMesh::~GeometryMesh() { }

/*! \param m Python module to export in
 */
void export_GeometryMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryMesh, Geometry, std::shared_ptr<GeometryMesh>>(m, "GeometryMesh")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int>())
        .def("getPositionBuffer", &GeometryMesh::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryMesh::getOrientationBuffer)
        .def("getColorBuffer", &GeometryMesh::getColorBuffer);
    }

    } // namespace gpu
    } // namespace fresnel
