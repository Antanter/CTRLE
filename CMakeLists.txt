cmake_minimum_required(VERSION 3.16)
project(LatencyAwareRL VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_FINDPYTHON ON)

set(CMAKE_BUILD_TYPE Release)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(robot_env python/envs/robot_env/robot_bindings.cpp)
target_link_libraries(robot_env PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_include_directories(robot_env PRIVATE include)
set_target_properties(robot_env PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)

target_compile_options(robot_env PRIVATE -march=native)

pybind11_add_module(racing_env python/envs/racing_env/racing_bindings.cpp)
target_link_libraries(racing_env PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_include_directories(racing_env PRIVATE include)
set_target_properties(racing_env PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)

target_compile_options(racing_env PRIVATE -march=native)
