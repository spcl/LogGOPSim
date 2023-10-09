cmake_minimum_required(VERSION 3.12)
project(LogGOPSim)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

include(${CMAKE_MODULE_PATH}/gengetopt.cmake)
include(${CMAKE_MODULE_PATH}/re2c.cmake)
find_gengetopt()
find_re2c()

# build loggopsim
find_package(Graphviz REQUIRED)
include_directories(${GRAPHVIZ_INCLUDE_DIRS})
add_gengetopt_files(LogGOPSim/loggopsim_cmdline)
add_executable(LogGOPSim LogGOPSim/loggopsim_cmdline.c LogGOPSim/LogGOPSim.cpp)
target_link_libraries(LogGOPSim ${GRAPHVIZ_CGRAPH_LIBRARY})

# build txt2bin
add_gengetopt_files(LogGOPSim/txt2bin_cmdline)
add_re2c_files(LogGOPSim/txt2bin)
add_executable(txt2bin LogGOPSim/txt2bin.cpp LogGOPSim/txt2bin_cmdline.c)

# build Schedgen1 (Schedgen2 is pure python)
add_gengetopt_files(Schedgen/schedgen_cmdline)
add_executable(schedgen Schedgen/buffer_element.cpp Schedgen/schedgen.cpp Schedgen/process_trace.cpp Schedgen/schedgen_cmdline.c)

# build liballprof2
find_package(MPI)
find_package(Clang)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_custom_target(mpi_header_avail      DEPENDS ${PROJECT_SOURCE_DIR}/liballprof2/mpi.h)
add_custom_target(mpi_wrapper_generated DEPENDS ${PROJECT_SOURCE_DIR}/liballprof2/mpi_wrapper.c)
add_custom_command(OUTPUT liballprof2/mpi_sem.yml
    COMMAND ${PROJECT_SOURCE_DIR}/liballprof2/gencode.py -s ${PROJECT_SOURCE_DIR}/liballprof2/mpi_sem.yml  DEPENDS mpi_header_avail)
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/liballprof2/mpi_wrapper.c
    COMMAND ${PROJECT_SOURCE_DIR}/liballprof2/gensem.py -l ${CLANG_LIBS} -m ${PROJECT_SOURCE_DIR}/liballprof2/mpi.h DEPENDS mpi_header_avail)
add_library(mpipclog liballprof2/mpi_c_wrapper.c liballprof2/mpi_f_wrapper.c)
add_dependencies(mpipclog mpi_wrapper_generated)
target_link_libraries(mpipclog ${MPI_C_LIBRARIES})


include(CTest)
# test the c++ toolchain for some pattern
add_test(NAME schedgen_binomialtreebcast COMMAND schedgen -o schedule.goal --commsize 8 --ptrn binomialtreebcast)
add_test(NAME txt2bin_binomialtreebcast COMMAND txt2bin -i schedule.goal -o schedule.bin)
add_test(NAME loggopsim_binomialtreebcast COMMAND LogGOPSim -f schedule.bin)

# test the python toolchain for some pattern
add_test(NAME schedgen2_bcast COMMAND ${PYTHON_EXECUTABLE} schedgen.py bcast --output schedule.goal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Schedgen2)
add_test(NAME txt2bin_schedgen2_bcast COMMAND txt2bin -i ${PROJECT_SOURCE_DIR}/Schedgen2/schedule.goal -o schedule.bin)
add_test(NAME loggopsim_schedgen2_bcast COMMAND LogGOPSim -f schedule.bin)

# test liballprof

# test liballprof2

