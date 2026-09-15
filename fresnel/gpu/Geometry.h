// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "cuda_platform.h"
#include <pybind11/pybind11.h>

#include "GeometryData.h"
#include "Scene.h"
#include "common/Material.h"

namespace fresnel
    {
namespace gpu
    {
//! Handle basic geometry methods
/*! Geometry tracks the geometry slots that belong to a given Scene. A Geometry object adds itself
   to the Scene on construction. When destructed, the Geometry is removed from the Scene. This
   requires that the user hold onto the Geometry shared pointer as long as they want to keep the
   object live in the scene. There is also an explicit remove function. When a given Geometry object
   is removed from a Scene, calls to change that geometry will fail.

    The base class Geometry itself does not define geometry. It just provides common methods and
   memory management. For derived classes, the bool value m_valid is true when the Geometry is added
   to the scene. Derived classes fill out m_data and set m_valid to true after attaching themselves
   to the Scene.

    A derived class owns the managed buffers its GeometryData points at, so the Scene's copy of the
   description is only valid while the Geometry is alive and attached.

    Each Geometry has a Material and an outline Material and an outline width, but these are managed
   by Scene. In the tracing kernel, only the scene and geometry id are available.
*/
class Geometry
    {
    public:
    //! Constructor
    Geometry(std::shared_ptr<Scene> scene);
    //! Destructor
    virtual ~Geometry();

    //! Enable the Geometry
    void enable();

    //! Disable the Geometry
    void disable();

    //! Remove the Geometry from the Scene
    void remove();

    //! Get the material
    const Material& getMaterial()
        {
        return m_scene->getMaterial(m_geom_id);
        }

    //! Get the outline material
    const Material& getOutlineMaterial()
        {
        return m_scene->getOutlineMaterial(m_geom_id);
        }

    //! Get the outline width
    float getOutlineWidth()
        {
        return m_scene->getOutlineWidth(m_geom_id);
        }

    //! Set the material
    void setMaterial(const Material& material)
        {
        m_scene->setMaterial(m_geom_id, material);
        }

    //! Set the outline material
    void setOutlineMaterial(const Material& material)
        {
        m_scene->setOutlineMaterial(m_geom_id, material);
        }

    //! Set the outline width
    void setOutlineWidth(float width)
        {
        m_scene->setOutlineWidth(m_geom_id, width);
        }

    //! Notify the geometry that changes have been made to the buffers
    /*! The buffers themselves are managed memory that the traversal reads in place, so this only
        republishes the description. The Scene rebuilds the acceleration structure on every render.
    */
    void update()
        {
        if (m_valid)
            m_scene->updateGeometry(m_geom_id, m_data);
        }

    protected:
    //! Attach the filled out description to the Scene
    void attach();

    unsigned int m_geom_id = 0; //!< Associated geometry id
    bool m_valid = false; //!< true when the geometry is valid and attached to the Scene
    std::shared_ptr<Scene> m_scene; //!< The scene the geometry is attached to
    std::shared_ptr<Device> m_device; //!< The device the Scene is attached to
    GeometryData m_data; //!< Device visible description of this geometry
    };

//! Export Geometry to python
void export_Geometry(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
