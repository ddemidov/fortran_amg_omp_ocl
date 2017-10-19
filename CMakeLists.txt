cmake_minimum_required(VERSION 3.1)
project(libamgcl)

add_subdirectory(vexcl)
add_subdirectory(amgcl)

add_library(libamgcl SHARED amgcl.cpp
    $<$<PLATFORM_ID:Windows>:dll.def>
    )

target_compile_definitions(libamgcl PRIVATE VEXCL_THROW_ON_EMPTY_CONTEXT)
target_link_libraries(libamgcl PRIVATE amgcl::amgcl VexCL::OpenCL)
set_target_properties(libamgcl PROPERTIES PREFIX "")

if (WIN32)
    set_target_properties(libamgcl PROPERTIES LINK_FLAGS "/DELAYLOAD:OpenCL.dll")
else()
    enable_language(Fortran)

    add_executable(poisson_omp poisson.F90 amgcl.F90)
    target_compile_definitions(poisson_omp PRIVATE DEVNUM=-1)
    target_link_libraries(poisson_omp libamgcl)

    add_executable(poisson_ocl poisson.F90 amgcl.F90)
    target_compile_definitions(poisson_ocl PRIVATE DEVNUM=0)
    target_link_libraries(poisson_ocl libamgcl)
endif()

