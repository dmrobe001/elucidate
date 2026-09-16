// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef SCENE_H_
#define SCENE_H_

#include "cuda_platform.h"
#include <pybind11/pybind11.h>
#include <vector>

#include "BVH.h"
#include "Device.h"
#include "GeometryData.h"
#include "common/Camera.h"
#include "common/Light.h"
#include "common/Material.h"

namespace fresnel
    {
namespace gpu
    {
//! Device visible view of a committed Scene
/*! Everything a tracer kernel needs to shade a hit: the hierarchy to traverse, the geometry
   descriptions the traversal switches on, and the per geometry id materials.
*/
struct SceneView
    {
    BVHView bvh; //!< The acceleration structure
    const GeometryData* geometry; //!< Geometry descriptions, indexed by geometry id
    const Material* materials; //!< Materials, indexed by geometry id
    const Material* outline_materials; //!< Outline materials, indexed by geometry id
    const float* outline_widths; //!< Outline widths, indexed by geometry id
    unsigned int n_geometry; //!< Number of geometry slots
    };

//! Container for the geometry and the acceleration structure built over it
/*! Handle the geometry registry and python lifetime as an exported class.

    Store the per geometry id materials in Scene. Geometry ids are small, increasing, and reused
   when geometry is deleted, so materials are stored in a std::vector. The vectors are mirrored into
   managed memory on commit() so that the tracer kernels can index them the same way.

    A given Scene also has an associated camera, background color, and background alpha. The camera
   is used by the Tracer to generate rays into the Scene. The background color and alpha are the
   resulting color output by the Tracer when a ray fails to hit geometry in the Scene.

    Scene will eventually support multiple lights. As a temporary API, Scene stores a single light
   direction.
*/
class Scene
    {
    public:
    //! Constructor
    Scene(std::shared_ptr<Device> device);
    //! Destructor
    ~Scene();

    //! Access the Device
    std::shared_ptr<Device> getDevice()
        {
        return m_device;
        }

    //! Add a geometry to the scene
    /*! \param data Description of the geometry
        \returns The geometry id assigned to it
    */
    unsigned int attachGeometry(const GeometryData& data);

    //! Update the description of an attached geometry
    void updateGeometry(unsigned int geom_id, const GeometryData& data);

    //! Remove a geometry from the scene
    void detachGeometry(unsigned int geom_id);

    //! Rebuild the acceleration structure and refresh the device side materials
    /*! This is the equivalent of rtcCommitScene(). The tracers call it on every render, and it
        rebuilds the hierarchy from scratch: python writes geometry buffers through a mapped numpy
        view, so there is no change notification to refit against.
    */
    void commit();

    //! Get a device visible view of the committed scene
    SceneView getSceneView() const;

    //! Set the material for a given geometry id
    void setMaterial(unsigned int geom_id, const Material& material)
        {
        if (geom_id >= m_materials.size())
            m_materials.resize(geom_id + 1);

        m_materials[geom_id] = material;
        }

    //! Get the material for a given geometry id
    const Material& getMaterial(unsigned int geom_id)
        {
        if (geom_id >= m_materials.size())
            m_materials.resize(geom_id + 1);

        return m_materials[geom_id];
        }

    //! Set the outline material for a given geometry id
    void setOutlineMaterial(unsigned int geom_id, const Material& material)
        {
        if (geom_id >= m_outline_materials.size())
            m_outline_materials.resize(geom_id + 1);

        m_outline_materials[geom_id] = material;
        }

    //! Get the outline material for a given geometry id
    const Material& getOutlineMaterial(unsigned int geom_id)
        {
        if (geom_id >= m_outline_materials.size())
            m_outline_materials.resize(geom_id + 1);

        return m_outline_materials[geom_id];
        }

    //! Set the outline width for a given geometry id
    void setOutlineWidth(unsigned int geom_id, float width)
        {
        if (geom_id >= m_outline_widths.size())
            m_outline_widths.resize(geom_id + 1);

        m_outline_widths[geom_id] = width;
        }

    //! Get the outline material for a given geometry id
    float getOutlineWidth(unsigned int geom_id)
        {
        if (geom_id >= m_outline_widths.size())
            m_outline_widths.resize(geom_id + 1);

        return m_outline_widths[geom_id];
        }

    //! Set the camera
    void setCamera(const UserCamera& camera)
        {
        m_camera = camera;
        }

    //! Get the camera
    UserCamera& getCamera()
        {
        return m_camera;
        }

    //! Set the background color
    void setBackgroundColor(const RGB<float>& c)
        {
        m_background_color = c;
        }

    //! Get the background color
    RGB<float> getBackgroundColor() const
        {
        return m_background_color;
        }

    //! Set the background alpha
    void setBackgroundAlpha(float a)
        {
        m_background_alpha = a;
        }

    //! Get the background alpha
    float getBackgroundAlpha() const
        {
        return m_background_alpha;
        }

    //! Get the lights
    Lights& getLights()
        {
        return m_lights;
        }

    //! Set the lights
    void setLights(const Lights& lights)
        {
        m_lights = lights;
        }

    private:
    std::shared_ptr<Device> m_device; //!< The device the scene is attached to

    std::vector<GeometryData> m_geometry; //!< Geometry descriptions, indexed by geometry id
    std::vector<unsigned int> m_free_ids; //!< Geometry ids released by detachGeometry()

    std::vector<Material> m_materials; //!< Materials associated with geometry ids
    std::vector<Material> m_outline_materials; //!< Materials associated with geometry ids
    std::vector<float> m_outline_widths; //!< Materials associated with geometry ids

    ManagedBuffer<GeometryData> m_d_geometry; //!< Device copy of m_geometry
    ManagedBuffer<Material> m_d_materials; //!< Device copy of m_materials
    ManagedBuffer<Material> m_d_outline_materials; //!< Device copy of m_outline_materials
    ManagedBuffer<float> m_d_outline_widths; //!< Device copy of m_outline_widths
    ManagedBuffer<unsigned int> m_d_offsets; //!< Prefix sum of the per geometry primitive counts

    BVH m_bvh; //!< The acceleration structure over every geometry

    RGB<float> m_background_color; //!< The background color
    float m_background_alpha; //!< Background alpha
    UserCamera m_camera; //!< The camera
    Lights m_lights; //!< The lights
    };

//! Export Scene to python
void export_Scene(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
