cmake_minimum_required(VERSION 3.14)
project(TensorView)

option(BUILD_TESTS "Build Google Test" ON)

set(CMAKE_CXX_STANDARD 14)

find_package(Boost REQUIRED)

set(TV_SOURCES
        TensorView/Traits.h
        TensorView/TensorView.h
        TensorView/TensorViewFwd.h
        TensorView/Operations.h
        TensorView/Utils.h
        TensorView/TensorIO.h
        TensorView/Dims.h
        )

add_library(TensorView INTERFACE)

target_include_directories(TensorView INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

add_executable(10sor main.cpp)
target_link_libraries(10sor PRIVATE TensorView)
target_include_directories(10sor PRIVATE Boost)

if (BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TensorView)

install(TARGETS TensorView ${IE_INSTALL_TARGETES}
        EXPORT tensor_view_export
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT tensor_view_export
        FILE TensorViewTargets.cmake
        NAMESPACE TV::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorView)

# Generate Config File
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/TensorViewConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/TensorViewConfig.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TensorViewConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

# export build tree
export(EXPORT tensor_view_export FILE ${CMAKE_CURRENT_BINARY_DIR}/TensorViewTargets.cmake NAMESPACE TV::)
export(PACKAGE TensorView)
