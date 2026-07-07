cmake_minimum_required(VERSION 3.20)

project(nccl_ollama_multigpu
        LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# Build Option
# -----------------------------------------------------------------------
option(BUILD_WITH_CUDA "Build CUDA/NCCL backend" ON)
option(BUILD_WITH_HIP  "Build HIP/RCCL backend" OFF)

# -----------------------------------------------------------------------
# MPI
# -----------------------------------------------------------------------
find_package(MPI REQUIRED)
message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
message(STATUS "MPI include dirs: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI libraries   : ${MPI_CXX_LIBRARIES}")

# -----------------------------------------------------------------------
# llama.cpp
# -----------------------------------------------------------------------
set(LLAMA_ROOT         "${CMAKE_SOURCE_DIR}/../llama.cpp")
set(LLAMA_INCLUDE_DIR  "${LLAMA_ROOT}/include")
set(LLAMA_LIBRARY_DIR  "${LLAMA_ROOT}/build")

find_library(LLAMA_LIB
    NAMES llama
    HINTS ${LLAMA_LIBRARY_DIR})

if (NOT LLAMA_LIB)
    message(WARNING "Could not find libllama in ${LLAMA_LIBRARY_DIR}. "
                    "Please adjust LLAMA_LIBRARY_DIR or build llama.cpp.")
else()
    message(STATUS "Found libllama: ${LLAMA_LIB}")
endif()

# -----------------------------------------------------------------------
# External includes (nlohmann/json, cpp-httplib)
# -----------------------------------------------------------------------
set(EXTERNAL_JSON_ROOT    "${CMAKE_SOURCE_DIR}/../json")
set(EXTERNAL_HTTPLIB_ROOT "${CMAKE_SOURCE_DIR}/../cpp-httplib")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external   
    ${EXTERNAL_JSON_ROOT}/single_include
    ${EXTERNAL_HTTPLIB_ROOT}
    ${MPI_CXX_INCLUDE_DIRS}
    ${LLAMA_INCLUDE_DIR}
)

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

# -----------------------------------------------------------------------
# Backend CUDA + NCCL
# -----------------------------------------------------------------------
if (BUILD_WITH_CUDA)
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries   : ${CUDAToolkit_LIBRARY_DIR}")

    find_library(NCCL_LIBRARY
        NAMES nccl
        HINTS
            ${CUDAToolkit_LIBRARY_DIR}
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /usr/local/cuda/lib64
    )

    if (NOT NCCL_LIBRARY)
        message(FATAL_ERROR "Could not find NCCL library (libnccl)")
    else()
        message(STATUS "NCCL library: ${NCCL_LIBRARY}")
    endif()

    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    add_executable(nccl_ollama_multigpu ${SRC_FILES})

    set_target_properties(nccl_ollama_multigpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(nccl_ollama_multigpu
        PRIVATE
            MPI::MPI_CXX
            CUDA::cudart
            ${NCCL_LIBRARY}
    )

    if (LLAMA_LIB)
        target_link_libraries(nccl_ollama_multigpu
            PRIVATE
                ${LLAMA_LIB}
        )
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(nccl_ollama_multigpu
            PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
    endif()
endif()

# -----------------------------------------------------------------------
# Backend HIP + RCCL
# -----------------------------------------------------------------------
if (BUILD_WITH_HIP)
    set(HIP_INCLUDE_DIR "/opt/rocm/include")
    include_directories(${HIP_INCLUDE_DIR})

    # RCCL (librccl)
    find_library(RCCL_LIBRARY
        NAMES rccl
        HINTS
            /opt/rocm/lib
            /opt/rocm/lib64
            /usr/lib
            /usr/lib64
    )

    if (NOT RCCL_LIBRARY)
        message(FATAL_ERROR "Could not find RCCL library (librccl)")
    else()
        message(STATUS "RCCL library: ${RCCL_LIBRARY}")
    endif()

    add_executable(Supra_AI_rccl_ollama_multigpu ${SRC_FILES})

    target_compile_definitions(Supra_AI_rccl_ollama_multigpu
        PRIVATE USE_HIP)

    target_link_libraries(Supra_AI_rccl_ollama_multigpu
        PRIVATE
            MPI::MPI_CXX
            ${RCCL_LIBRARY}
    )

    if (LLAMA_LIB)
        target_link_libraries(Supra_AI_rccl_ollama_multigpu
            PRIVATE
                ${LLAMA_LIB}
        )
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(Supra_AI_rccl_ollama_multigpu
            PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
    endif()
endif()