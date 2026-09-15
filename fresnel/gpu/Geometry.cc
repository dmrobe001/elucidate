// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Geometry.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    The base class constructor does nothing beyond zeroing the description. It is up to the derived
   classes to fill out m_data, call attach(), and set m_valid.
*/
Geometry::Geometry(std::shared_ptr<Scene> scene) : m_scene(scene), m_device(scene->getDevice())
    {
    // value initialization leaves every buffer pointer null and every count zero; a derived
    // class fills in only the fields its primitive type uses
    m_data = GeometryData();
    m_data.enabled = 1;
    }

Geometry::~Geometry()
    {
    remove();
    }

/*! Hand the filled out description to the Scene and claim a geometry id.
 */
void Geometry::attach()
    {
    m_geom_id = m_scene->attachGeometry(m_data);
    m_valid = true;
    }

/*! When enabled, the geometry will be present when rendering the scene
 */
void Geometry::enable()
    {
    if (m_valid)
        {
        m_data.enabled = 1;
        m_scene->updateGeometry(m_geom_id, m_data);
        }
    else
        {
        throw std::runtime_error("Cannot enable inactive Geometry");
        }
    }

/*! When disabled, the geometry will not be present in the scene. No rays will intersect it.
 */
void Geometry::disable()
    {
    if (m_valid)
        {
        m_data.enabled = 0;
        m_scene->updateGeometry(m_geom_id, m_data);
        }
    else
        {
        throw std::runtime_error("Cannot disable inactive Geometry");
        }
    }

/*! Once it is removed from a Scene, the Geometry cannot be changed, enabled, or disabled.
    remove() may be called multiple times. It has no effect on subsequent calls.
*/
void Geometry::remove()
    {
    if (m_valid)
        {
        m_scene->detachGeometry(m_geom_id);
        m_valid = false;
        }
    }

/*! \param m Python module to export in
 */
void export_Geometry(pybind11::module& m)
    {
    pybind11::class_<Geometry, std::shared_ptr<Geometry>>(m, "Geometry")
        .def(pybind11::init<std::shared_ptr<Scene>>())
        .def("getMaterial", &Geometry::getMaterial)
        .def("setMaterial", &Geometry::setMaterial)
        .def("getOutlineMaterial", &Geometry::getOutlineMaterial)
        .def("setOutlineMaterial", &Geometry::setOutlineMaterial)
        .def("getOutlineWidth", &Geometry::getOutlineWidth)
        .def("setOutlineWidth", &Geometry::setOutlineWidth)
        .def("disable", &Geometry::disable)
        .def("enable", &Geometry::enable)
        .def("remove", &Geometry::remove)
        .def("update", &Geometry::update);
    }

    } // namespace gpu
    } // namespace fresnel
