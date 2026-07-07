#pragma once

#include <nccl.h>
#include <stdexcept>
#include <string>

// Simple wrapper for NCCL error checking
inline void checkNccl(ncclResult_t res, const char* msg) {
    if (res != ncclSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + ncclGetErrorString(res));
    }
}

// Initialize a NCCL communicator from a ncclUniqueId.
//
// Typically, rank 0 gets the id via ncclGetUniqueId and broadcasts it via MPI.
void init_nccl_comm(ncclComm_t* comm,
                    const ncclUniqueId* id,
                    int world_size,
                    int world_rank);

// Destroy a NCCL communicator (wrapped for safety).
void destroy_nccl_comm(ncclComm_t comm);