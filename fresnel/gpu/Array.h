// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef ARRAY_H_
#define ARRAY_H_

#include "cuda_platform.h"

#include <pybind11/pybind11.h>

namespace fresnel
    {
namespace gpu
    {
namespace detail
    {
template<class T> unsigned int array_width(const T& a)
    {
    return 1;
    }

template<class T> std::string array_dtype(const T& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const vec2<T>& a)
    {
    return 2;
    }

template<class T> std::string array_dtype(const vec2<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const vec3<T>& a)
    {
    return 3;
    }

template<class T> std::string array_dtype(const vec3<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const quat<T>& a)
    {
    return 4;
    }

template<class T> std::string array_dtype(const quat<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const RGBA<T>& a)
    {
    return 4;
    }

template<class T> std::string array_dtype(const RGBA<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const RGB<T>& a)
    {
    return 3;
    }

template<class T> std::string array_dtype(const RGB<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }
    } // namespace detail

//! Managed memory buffer
/*! Allocate \a n elements of \a T with cudaMallocManaged and zero them. The single allocation is
   reachable from both the host and the kernels, which is what lets the Scene and its geometry be
   assembled by ordinary host code and then read during traversal without an explicit copy step.

    ManagedBuffer is for internal storage. User facing buffers use Array, which adds the numpy
   buffer protocol on top of the same allocation.
*/
template<class T> class ManagedBuffer
    {
    public:
    //! Allocate an empty buffer
    ManagedBuffer() : m_data(nullptr), m_size(0) { }

    //! Allocate \a n elements
    explicit ManagedBuffer(size_t n) : m_data(nullptr), m_size(n)
        {
        allocate(n);
        }

    ~ManagedBuffer()
        {
        // a destructor may not throw, and there is nothing to be done about a failed free
        if (m_data != nullptr)
            cudaFree(m_data);
        }

    // the buffer owns a device allocation, so it may not be copied
    ManagedBuffer(const ManagedBuffer&) = delete;
    ManagedBuffer& operator=(const ManagedBuffer&) = delete;

    //! Discard the contents and allocate \a n elements
    void resize(size_t n)
        {
        if (n == m_size)
            return;

        if (m_data != nullptr)
            {
            CUDA_CHECK(cudaFree(m_data));
            m_data = nullptr;
            }

        m_size = n;
        allocate(n);
        }

    //! Number of elements in the buffer
    size_t size() const
        {
        return m_size;
        }

    //! Access the managed pointer
    T* get() const
        {
        return m_data;
        }

    //! Host element access
    /*! The caller is responsible for synchronizing with any in-flight kernel.
     */
    T& operator[](size_t i) const
        {
        return m_data[i];
        }

    private:
    void allocate(size_t n)
        {
        if (n == 0)
            return;

        CUDA_CHECK(cudaMallocManaged(&m_data, sizeof(T) * n));
        CUDA_CHECK(cudaMemset(m_data, 0, sizeof(T) * n));

        // the memset leaves primitive types zeroed; run the constructors for the rest
        for (size_t i = 0; i < n; ++i)
            ::new ((void*)&m_data[i]) T;
        }

    T* m_data; //!< Managed allocation
    size_t m_size; //!< Number of elements
    };

//! Array data
/*! Define an array data structure

    This array class encapsulates a 1 or 2 dimensional array of data elements. It is designed to be
   an abstract way of accessing and storing data that is available to the CUDA kernels from python.
   It works with a proxy class written in python to allow map/unmap semantics implicitly while the
   user sees a numpy-like interface.

    Array should be used for python-facing data structures that the user modifies directly. Internal
   storage buffers should be kept in whatever internal storage is appropriate - use of the Array
   class signifies that this will be directly user-accessible.

    Arrays of vector types (vec3, RGBA, etc...) automatically map to WxHx3 (or 4) numpy arrays to
   allow users natural access to the individual data elements from within python.

    The storage is managed memory, so the numpy view handed to python points at the same allocation
   the kernels read. map() synchronizes the device first: managed memory may not be touched by the
   host while a kernel that could reach it is still running.
*/
template<class T> class Array
    {
    public:
    //! Default constructor
    Array()
        {
        m_w = m_h = 0;
        m_ndim = 1;
        }

    //! Construct a 1D array
    Array(size_t n) : m_data(n)
        {
        m_w = n;
        m_h = 1;
        m_ndim = 1;
        }

    //! Construct a 2D array
    Array(size_t w, size_t h) : m_data(w * h)
        {
        m_w = w;
        m_h = h;
        m_ndim = 2;
        }

    //! Get a python buffer pointing to the data
    pybind11::buffer_info getBuffer()
        {
        std::vector<size_t> shape;
        std::vector<size_t> strides;

        unsigned int array_width = detail::array_width(T());
        size_t item_size = sizeof(T) / array_width;

        //! build up the shape and strides arrays
        if (m_ndim == 1)
            {
            if (array_width == 1)
                {
                shape = {m_w};
                strides = {item_size};
                }
            else
                {
                shape = {m_w, array_width};
                strides = {item_size * array_width, item_size};
                }
            }
        else
            {
            if (array_width == 1)
                {
                shape = {m_h, m_w};
                strides = {m_w * item_size, item_size};
                }
            else
                {
                shape = {m_h, m_w, array_width};
                strides = {m_w * item_size * array_width, item_size * array_width, item_size};
                }
            }

        unsigned int dim = m_ndim;
        if (array_width > 1)
            dim += 1;

        return pybind11::buffer_info(m_data.get(),
                                     item_size,
                                     detail::array_dtype(T()),
                                     dim,
                                     shape,
                                     strides);
        }

    //! Get the width of the array
    size_t getW()
        {
        return m_w;
        }

    //! Get the height of the array
    size_t getH()
        {
        return m_h;
        }

    //! Get the number of dimensions in the array
    unsigned int getNDim()
        {
        return m_ndim;
        }

    //! Data accessor
    const T& get(size_t i) const
        {
        return m_data[i];
        }

    //! Get the device visible pointer
    /*! Unlike map(), this does not synchronize. It is for handing storage to a kernel, not for
        reading it from the host.
    */
    T* getPointer() const
        {
        return m_data.get();
        }

    //! Bind the array
    T* map()
        {
        CUDA_CHECK(cudaDeviceSynchronize());
        return m_data.get();
        }

    //! Map from python
    void map_py()
        {
        // it is important that the python mapping method does not return a pointer
        // if you return a bare pointer with pybind11, pybind11 will try to free it!
        CUDA_CHECK(cudaDeviceSynchronize());
        }

    //! Unbind the array
    void unmap() { }

    protected:
    ManagedBuffer<T> m_data; //!< Stored data
    size_t m_w; //!< Width of data array
    size_t m_h; //!< Height of data array
    unsigned int m_ndim; //!< Number of dimensions in the data array
    };

//! Export Array instantiations to python
void export_Array(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
