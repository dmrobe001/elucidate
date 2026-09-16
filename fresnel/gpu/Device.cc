// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "Device.h"

using namespace std;

namespace fresnel
    {
namespace gpu
    {
/*! \returns The number of CUDA devices present, or 0 when the CUDA runtime cannot be initialized.

    A missing driver is not an error: fresnel reports no GPUs and python falls back to the CPU
   backend. Both `_gpu.get_num_available_devices()` and `Device.getAllGPUs()` are called at import
   time, so neither may throw on a machine without a GPU.
*/
int Device::getNumAvailableDevices()
    {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
        {
        // clear the sticky error so that a later, legitimate call is not misdiagnosed
        cudaGetLastError();
        return 0;
        }

    return count;
    }

/*! Select the CUDA devices to render on.

    \param n Number of GPUs to use (-1 to use all).
*/
Device::Device(int n)
    {
    int count = getNumAvailableDevices();

    int n_to_use = count;
    if (n != -1)
        n_to_use = n;

    if (n_to_use > count)
        n_to_use = count;

    if (n_to_use < 1)
        throw std::runtime_error("No CUDA devices are available");

    for (int i = 0; i < n_to_use; i++)
        m_devices.push_back(i);

    // fail here rather than at the first launch when the device cannot be initialized
    CUDA_CHECK(cudaSetDevice(m_devices[0]));
    CUDA_CHECK(cudaFree(0));
    }

Device::~Device() { }

//! Format a device list the way the OptiX backend used to
static std::string formatDeviceList(const std::vector<int>& devices)
    {
    ostringstream s;

    for (const int& i : devices)
        {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        float ghz = float(prop.clockRate) / 1e6f;
        int mib = int(float(prop.totalGlobalMem) / (1024.0f * 1024.0f));

        s << " [" + to_string(i) + "]: " << setw(22) << prop.name << " " << setw(4)
          << prop.multiProcessorCount << " " << "SM_" << prop.major << "." << prop.minor << " ";
        s.precision(3);
        s.fill('0');

        s << "@ " << setw(4) << ghz << " GHz";
        s.fill(' ');
        s << ", " << setw(5) << mib << " MiB DRAM" << std::endl;
        }

    return s.str();
    }

/*! \returns Human readable string containing useful device information
 */
std::string Device::describe() const
    {
    return "Enabled CUDA devices:\n" + formatDeviceList(m_devices);
    }

/*! \returns Human readable string listing every CUDA device on the system

    fresnel/__init__.py splits this on newlines to fill `Device.available_gpus`, so every device
   must occupy exactly one line.
*/
std::string Device::getAllGPUs()
    {
    int count = getNumAvailableDevices();

    vector<int> devices;
    for (int i = 0; i < count; i++)
        devices.push_back(i);

    return formatDeviceList(devices);
    }

/*! \param m Python module to export in
 */
void export_Device(pybind11::module& m)
    {
    pybind11::class_<Device, std::shared_ptr<Device>>(m, "Device")
        .def(pybind11::init<int>())
        .def("describe", &Device::describe)
        .def_static("getAllGPUs", &Device::getAllGPUs);

    m.def("get_num_available_devices", &Device::getNumAvailableDevices);
    }

    } // namespace gpu
    } // namespace fresnel
