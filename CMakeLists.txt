cmake_minimum_required(VERSION 3.16)
project(pelletcz LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(PELLETCZ_USE_OPENMP "Thread-parallel assembly and solvers (OpenMP)" ON)
option(PELLETCZ_USE_MPI    "MPI ensemble driver (Weibull realisations / mesh sweeps)" ON)
option(PELLETCZ_NATIVE     "Compile with -march=native" OFF)
option(PELLETCZ_BUILD_TESTS "Build unit tests" ON)

add_library(pelletcz_core
  src/config.cpp
  src/mesh.cpp
  src/sparse.cpp
  src/amg.cpp
  src/random_field.cpp
  src/materials.cpp
  src/interface_law.cpp
  src/model.cpp
  src/thermal.cpp
  src/mechanics.cpp
  src/solver.cpp
  src/output.cpp
)
target_include_directories(pelletcz_core PUBLIC src)

# build id = sha256 of the solver sources (+ git revision when available), recorded in every manifest
file(GLOB PELLETCZ_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
list(SORT PELLETCZ_SOURCES)
set(_cat "")
foreach(f ${PELLETCZ_SOURCES})
  file(SHA256 ${f} _h)
  string(APPEND _cat ${_h})
endforeach()
string(SHA256 _src_hash "${_cat}")
string(SUBSTRING ${_src_hash} 0 12 _src_hash)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE _git OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_git)
  set(PELLETCZ_BUILD_ID "src-${_src_hash}-git-${_git}")
else()
  set(PELLETCZ_BUILD_ID "src-${_src_hash}")
endif()
target_compile_definitions(pelletcz_core PRIVATE PELLETCZ_BUILD_ID="${PELLETCZ_BUILD_ID}")
target_compile_options(pelletcz_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(PELLETCZ_NATIVE)
  target_compile_options(pelletcz_core PUBLIC -march=native)
endif()

if(PELLETCZ_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(pelletcz_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "pelletcz: OpenMP enabled")
  else()
    message(WARNING "pelletcz: OpenMP requested but not found - building serial")
  endif()
endif()

add_executable(pelletcz src/main.cpp)
target_link_libraries(pelletcz PRIVATE pelletcz_core)

if(PELLETCZ_USE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    target_link_libraries(pelletcz PRIVATE MPI::MPI_CXX)
    target_compile_definitions(pelletcz PRIVATE PELLETCZ_HAVE_MPI=1)
    message(STATUS "pelletcz: MPI ensemble driver enabled")
  else()
    message(STATUS "pelletcz: MPI not found - ensemble mode runs cases sequentially")
  endif()
endif()

if(PELLETCZ_BUILD_TESTS)
  enable_testing()
  add_executable(pelletcz_tests tests/test_main.cpp)
  target_link_libraries(pelletcz_tests PRIVATE pelletcz_core)
  add_test(NAME unit COMMAND pelletcz_tests)
endif()
