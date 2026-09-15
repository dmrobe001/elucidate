// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include "cuda_platform.h"
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include "Array.h" // not used by device, but for the pybind11 shared pointer holder type definition

namespace fresnel
    {
namespace gpu
    {
//! Thin wrapper for a set of CUDA devices
/*! Handle selection of the CUDA devices to render on, and python lifetime as an exported class.

    Rendering is issued on the first selected device. The remaining entries in the selection are
   reported by describe() so that users can see which GPUs a `fresnel.Device` claimed, mirroring
   `Device.available_gpus` at the python level.

    All buffers in this backend are allocated with cudaMallocManaged, so the Scene, its geometry,
   and the Tracer output are reachable from the host without explicit copies.
*/
class Device
    {
    public:
    //! Constructor
    Device(int n);

    //! Destructor
    ~Device();

    //! Get the CUDA device ordinal that kernels are launched on
    int getCUDADevice() const
        {
        return m_devices[0];
        }

    //! Make the render device current on the calling thread
    void makeCurrent() const
        {
        CUDA_CHECK(cudaSetDevice(m_devices[0]));
        }

    //! Get information about this device
    std::string describe() const;

    //! List all GPUs
    static std::string getAllGPUs();

    //! Number of CUDA devices that fresnel can render on
    static int getNumAvailableDevices();

    private:
    std::vector<int> m_devices; //!< Ordinals of the selected CUDA devices
    };

//! Export Device to python
void export_Device(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
