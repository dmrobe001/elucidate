// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Scene.h"

namespace fresnel
    {
namespace gpu
    {
//! An empty geometry slot, skipped by the traversal
static GeometryData emptyGeometry()
    {
    GeometryData data = {};
    data.type = geometry_sphere;
    data.n_primitives = 0;
    data.enabled = 0;
    return data;
    }

/*! \param device Device to attach the Scene to
 */
Scene::Scene(std::shared_ptr<Device> device)
    : m_device(device), m_background_color(RGB<float>(0, 0, 0)), m_background_alpha(0.0)
    {
    m_lights.N = 2;
    m_lights.direction[0] = vec3<float>(-1, 0.3, 1);
    m_lights.color[0] = RGB<float>(1, 1, 1);

    m_lights.direction[1] = vec3<float>(1, 0, 1);
    m_lights.color[1] = RGB<float>(0.1, 0.1, 0.1);
    }

Scene::~Scene() { }

/*! \param data Description of the geometry
    \returns The geometry id assigned to it

    Geometry ids are reused after detachGeometry() so that they stay small and dense, which is what
   lets Scene index materials with a std::vector.
*/
unsigned int Scene::attachGeometry(const GeometryData& data)
    {
    unsigned int geom_id;

    if (!m_free_ids.empty())
        {
        geom_id = m_free_ids.back();
        m_free_ids.pop_back();
        m_geometry[geom_id] = data;
        }
    else
        {
        geom_id = (unsigned int)m_geometry.size();
        m_geometry.push_back(data);
        }

    return geom_id;
    }

/*! \param geom_id Geometry to update
    \param data New description
*/
void Scene::updateGeometry(unsigned int geom_id, const GeometryData& data)
    {
    if (geom_id >= m_geometry.size())
        throw std::runtime_error("Invalid geometry id");

    m_geometry[geom_id] = data;
    }

/*! \param geom_id Geometry to remove

    The slot is emptied rather than erased: later geometry ids must not shift, because the materials
   are indexed by them.
*/
void Scene::detachGeometry(unsigned int geom_id)
    {
    if (geom_id >= m_geometry.size())
        return;

    m_geometry[geom_id] = emptyGeometry();
    m_free_ids.push_back(geom_id);
    }

/*! Mirror the host side geometry and materials into managed memory and rebuild the BVH.
 */
void Scene::commit()
    {
    m_device->makeCurrent();

    const unsigned int n_geometry = (unsigned int)m_geometry.size();

    // the materials are sized lazily by the setters, so a geometry may not have reached them yet
    if (m_materials.size() < n_geometry)
        m_materials.resize(n_geometry);
    if (m_outline_materials.size() < n_geometry)
        m_outline_materials.resize(n_geometry);
    if (m_outline_widths.size() < n_geometry)
        m_outline_widths.resize(n_geometry);

    // the host is about to write buffers the last render read
    CUDA_CHECK(cudaDeviceSynchronize());

    m_d_geometry.resize(n_geometry);
    m_d_materials.resize(n_geometry);
    m_d_outline_materials.resize(n_geometry);
    m_d_outline_widths.resize(n_geometry);
    m_d_offsets.resize(n_geometry + 1);

    m_d_offsets[0] = 0;
    for (unsigned int i = 0; i < n_geometry; i++)
        {
        m_d_geometry[i] = m_geometry[i];
        m_d_materials[i] = m_materials[i];
        m_d_outline_materials[i] = m_outline_materials[i];
        m_d_outline_widths[i] = m_outline_widths[i];
        m_d_offsets[i + 1] = m_d_offsets[i] + m_geometry[i].n_primitives;
        }

    m_bvh.build(m_d_geometry.get(), m_d_offsets.get(), n_geometry);
    }

SceneView Scene::getSceneView() const
    {
    SceneView view;
    view.bvh = m_bvh.view();
    view.geometry = m_d_geometry.get();
    view.materials = m_d_materials.get();
    view.outline_materials = m_d_outline_materials.get();
    view.outline_widths = m_d_outline_widths.get();
    view.n_geometry = (unsigned int)m_d_geometry.size();
    return view;
    }

/*! \param m Python module to export in
 */
void export_Scene(pybind11::module& m)
    {
    pybind11::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
        .def(pybind11::init<std::shared_ptr<Device>>())
        .def("setCamera", &Scene::setCamera)
        .def("getCamera", &Scene::getCamera, pybind11::return_value_policy::reference_internal)
        .def("getBackgroundColor", &Scene::getBackgroundColor)
        .def("setBackgroundColor", &Scene::setBackgroundColor)
        .def("getBackgroundAlpha", &Scene::getBackgroundAlpha)
        .def("setBackgroundAlpha", &Scene::setBackgroundAlpha)
        .def("getLights", &Scene::getLights, pybind11::return_value_policy::reference_internal)
        .def("setLights", &Scene::setLights);
    }

    } // namespace gpu
    } // namespace fresnel
