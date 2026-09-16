include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# variables needed for multi-config builds
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
file(RELATIVE_PATH relative_dir ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (_is_multi_config AND ${CMAKE_VERSION} VERSION_LESS "3.20.0")
    message(WARNING "Install fresnel to obtain a functioning Python package.\n"
                    "Use CMake 3.20 or newer to assemble a the Python package in ${PROJECT_BINARY_DIR}/<CONFIG>")
endif()

#####################################################
## C++ libraries

# _common
set(_common_sources
     common/module-common.cc
     common/ConvexPolyhedronBuilder.cc
    )

find_package(Qhull)

# work around broken libqhull installations
if (qhull_EMBED_SOURCE)
    list(APPEND _common_sources ${libqhullr_SOURCES} ${libqhullcpp_SOURCES})
endif()

pybind11_add_module(_common ${_common_sources})
if (_is_multi_config)
    set_target_properties(_common PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/$<CONFIG>/${relative_dir})
endif()
target_link_libraries(_common PRIVATE QHull::qhull_r QHull::qhull_cpp)
install(TARGETS _common LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})

# _cpu
if (ENABLE_EMBREE)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_CPU)

if (WIN32)
target_compile_definitions(_common PUBLIC _USE_MATH_DEFINES NOMINMAX R123_NO_SINCOS)
endif()

set(_cpu_sources
     cpu/module-cpu.cc
     cpu/Device.cc
     cpu/Scene.cc
     cpu/Geometry.cc
     cpu/GeometryCylinder.cc
     cpu/GeometryPolygon.cc
     cpu/GeometryMesh.cc
     cpu/GeometryConvexPolyhedron.cc
     cpu/GeometrySphere.cc
     cpu/Tracer.cc
     cpu/TracerDirect.cc
     cpu/TracerPath.cc
     cpu/Array.cc
   )

pybind11_add_module(_cpu "${_cpu_sources}")
target_link_libraries(_cpu PRIVATE embree TBB::tbb)
if (_is_multi_config)
    set_target_properties(_cpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/$<CONFIG>/${relative_dir})
endif()

if(APPLE)
    set_target_properties(_cpu PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../../..")
else()
    set_target_properties(_cpu PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..")
endif()

if (WIN32)
target_compile_definitions(_cpu PUBLIC _USE_MATH_DEFINES NOMINMAX R123_NO_SINCOS)
endif()

install(TARGETS _cpu LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})

endif()

# _gpu
if (ENABLE_CUDA)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_GPU)

# host side: the device, buffers, geometry registry and pybind11 glue
set(_gpu_sources
     gpu/module-gpu.cc
     gpu/Device.cc
     gpu/Scene.cc
     gpu/Geometry.cc
     gpu/GeometryCylinder.cc
     gpu/GeometryPolygon.cc
     gpu/GeometryMesh.cc
     gpu/GeometryConvexPolyhedron.cc
     gpu/GeometrySphere.cc
     gpu/Tracer.cc
     gpu/Array.cc
   )

# device side: the acceleration structure build and the two render kernels
set(_gpu_cuda_sources
     gpu/BVH.cu
     gpu/TracerDirect.cu
     gpu/TracerPath.cu
   )

pybind11_add_module(_gpu "${_gpu_sources}" "${_gpu_cuda_sources}")
target_link_libraries(_gpu PRIVATE CUDA::cudart)

# link time optimization cannot see through the device code nvcc emits
set_target_properties(_gpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

if (_is_multi_config)
    set_target_properties(_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/$<CONFIG>/${relative_dir})
endif()

if(APPLE)
    set_target_properties(_gpu PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../../..")
else()
    set_target_properties(_gpu PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..")
endif()

if (WIN32)
target_compile_definitions(_gpu PUBLIC _USE_MATH_DEFINES NOMINMAX R123_NO_SINCOS)

# nvccfeatures.h hardcodes R123_USE_MULHILO64_ASM=1 for the host-side compile pass on the
# assumption that nvcc always falls through to gcc; on Windows it falls through to MSVC, which
# cannot compile GNU inline asm. Leaving the macro undefined here lets msvcfeatures.h's own
# MSVC_INTRIN selection take effect instead.
target_compile_definitions(_gpu PUBLIC R123_USE_MULHILO64_ASM=0)
endif()

install(TARGETS _gpu LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})

endif()

#####################################################
### Python only modules

set(files __init__.py
          conftest.py
          geometry.py
          tracer.py
          material.py
          camera.py
          util.py
          color.py
          light.py
          interact.py
          version.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
       )

copy_files_to_build("${files}" "fresnel" "*.py")

add_subdirectory(test)
