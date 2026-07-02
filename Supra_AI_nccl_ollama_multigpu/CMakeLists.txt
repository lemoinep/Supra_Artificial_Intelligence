# Update for my repo

cmake_minimum_required(VERSION 3.20)

project(Supra_AI_nccl_ollama_multigpu
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_WITH_CUDA "Build with CUDA support" ON)

find_package(MPI REQUIRED)
message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
message(STATUS "MPI include dirs: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI libraries   : ${MPI_CXX_LIBRARIES}")

# -----------------------------------------------------------------------
# External includes (json, cpp-httplib)
# -----------------------------------------------------------------------
set(EXTERNAL_JSON_ROOT "${CMAKE_SOURCE_DIR}/../json")
set(EXTERNAL_HTTPLIB_ROOT "${CMAKE_SOURCE_DIR}/../cpp-httplib")

include_directories(
    ${EXTERNAL_JSON_ROOT}/single_include
    ${EXTERNAL_HTTPLIB_ROOT}
    ${EXTERNAL_HTTPLIB_ROOT}
    ${MPI_CXX_INCLUDE_DIRS}
    ${LLAMA_INCLUDE_DIR}
)

# -----------------------------------------------------------------------
# llama.cpp 
# -----------------------------------------------------------------------

set(LLAMA_ROOT "${CMAKE_SOURCE_DIR}/../llama.cpp")
set(LLAMA_INCLUDE_DIR "${LLAMA_ROOT}/include")
set(LLAMA_LIBRARY_DIR "${LLAMA_ROOT}/build")

find_library(LLAMA_LIB
    NAMES llama
    HINTS ${LLAMA_LIBRARY_DIR})

if (NOT LLAMA_LIB)
    message(WARNING "Could not find libllama. Please adjust LLAMA_LIBRARY_DIR.")
else()
    message(STATUS "Found libllama: ${LLAMA_LIB}")
endif()

# -----------------------------------------------------------------------
# CUDA & NCCL
# -----------------------------------------------------------------------
if (BUILD_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_library(NCCL_LIBRARY
        NAMES nccl
        HINTS ${CUDAToolkit_LIBRARY_DIR}
              /usr/lib/x86_64-linux-gnu
              /usr/local/lib
              /usr/local/cuda/lib64
    )

    if (NOT NCCL_LIBRARY)
        message(FATAL_ERROR "Could not find NCCL library")
    else()
        message(STATUS "NCCL library: ${NCCL_LIBRARY}")
    endif()
endif()

# -----------------------------------------------------------------------
# Project includes
# -----------------------------------------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${MPI_CXX_INCLUDE_DIRS}
    ${LLAMA_INCLUDE_DIR}
)

if (BUILD_WITH_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# -----------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------
set(SRC_FILES
    src/gpu_utils.cpp
    src/nccl_utils.cpp
    src/config.cpp
    src/llama_backend.cpp
    src/scheduler.cpp
    src/scheduler_threadsafe.cpp
    src/model_topology.cpp
    src/lora_nccl.cpp
    src/worker.cpp
    src/api_http.cpp
    src/main_mpi_nccl.cpp
)

add_executable(Supra_AI_nccl_ollama_multigpu ${SRC_FILES})

if (BUILD_WITH_CUDA)
    set_target_properties(Supra_AI_nccl_ollama_multigpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(Supra_AI_nccl_ollama_multigpu
    PRIVATE
        MPI::MPI_CXX
)

if (BUILD_WITH_CUDA)
    target_link_libraries(Supra_AI_nccl_ollama_multigpu
        PRIVATE
            CUDA::cudart
            ${NCCL_LIBRARY}
    )
endif()

if (LLAMA_LIB)
    target_link_libraries(Supra_AI_nccl_ollama_multigpu
        PRIVATE
            ${LLAMA_LIB}
    )
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Supra_AI_nccl_ollama_multigpu
        PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()