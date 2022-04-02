cmake_minimum_required(VERSION 3.10)

project(Leticia)

add_library(leticia SHARED leticia.cc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(leticia PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS leticia DESTINATION lib)
