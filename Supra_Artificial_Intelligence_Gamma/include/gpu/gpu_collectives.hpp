#pragma once

#include <nccl.h>

#include <stdexcept>
#include <string>

// =============================================================================
// GPU collective communication interface.
//
// Same public API for:
//   - NVIDIA CUDA + NCCL
//   - AMD HIP + RCCL
//
// RCCL exposes an NCCL-compatible API: ncclComm_t, ncclUniqueId,
// ncclCommInitRank, ncclCommDestroy, ncclBroadcast, etc.
// =============================================================================

// Throws a C++ exception if an NCCL/RCCL operation fails.
inline void check_collective(ncclResult_t result, const char* what) {
    if (result != ncclSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + ncclGetErrorString(result));
    }
}

// Initializes a communicator rank.
//
// `comm`       : output communicator.
// `id`         : unique ID created by the root with ncclGetUniqueId() and
//                shared with every participating rank (for example MPI_Bcast).
// `group_size` : number of participating ranks in this model group.
// `local_rank` : rank inside this model group, in [0, group_size).
//
// The current GPU device must already have been selected before calling this.
void init_gpu_comm(ncclComm_t*        comm,
                   const ncclUniqueId* id,
                   int                group_size,
                   int                local_rank);

// Safely destroys a communicator. Does nothing if comm == nullptr.
void destroy_gpu_comm(ncclComm_t comm);