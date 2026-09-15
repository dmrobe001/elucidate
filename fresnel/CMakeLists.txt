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
