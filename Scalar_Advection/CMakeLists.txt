cmake_minimum_required(VERSION 3.20)

project(scalar_advection)

find_package(samurai)
find_package(CLI11)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

if(PETSC_FOUND)
    find_package(MPI)
endif()

set(CMAKE_CXX_STANDARD 17)

add_executable(scalar_advection main.cpp)
target_link_libraries(scalar_advection samurai::samurai CLI11::CLI11)
