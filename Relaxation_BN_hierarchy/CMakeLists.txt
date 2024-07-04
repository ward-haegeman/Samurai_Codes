cmake_minimum_required(VERSION 3.20)

project(relaxation)

find_package(samurai)
find_package(CLI11)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

if(PETSC_FOUND)
    find_package(MPI)
endif()

set(CMAKE_CXX_STANDARD 17)

add_executable(relaxation main.cpp)
target_link_libraries(relaxation samurai::samurai CLI11::CLI11)
