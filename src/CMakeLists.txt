cmake_minimum_required(VERSION 3.12)
project(LogGOPSim)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

include(${CMAKE_MODULE_PATH}/gengetopt.cmake)
include(${CMAKE_MODULE_PATH}/re2c.cmake)

find_gengetopt()
find_re2c()
find_package(Graphviz REQUIRED)
include_directories(${GRAPHVIZ_INCLUDE_DIRS})

add_gengetopt_files(LogGOPSim/loggopsim_cmdline)
add_executable(LogGOPSim LogGOPSim/loggopsim_cmdline.c LogGOPSim/LogGOPSim.cpp)
target_link_libraries(LogGOPSim ${GRAPHVIZ_CGRAPH_LIBRARY})

add_gengetopt_files(LogGOPSim/txt2bin_cmdline)
add_re2c_files(LogGOPSim/txt2bin)
add_executable(txt2bin LogGOPSim/txt2bin.cpp LogGOPSim/txt2bin_cmdline.c)

add_gengetopt_files(Schedgen/schedgen_cmdline)
add_executable(schedgen Schedgen/buffer_element.cpp Schedgen/schedgen.cpp Schedgen/process_trace.cpp Schedgen/schedgen_cmdline.c)

