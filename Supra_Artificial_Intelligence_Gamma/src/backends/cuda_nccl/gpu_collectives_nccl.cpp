#include "gpu/gpu_collectives.hpp"

#include <stdexcept>

void init_gpu_comm(ncclComm_t* comm,
                   const ncclUniqueId* id,
                   int group_size,
                   int local_rank) {
    if (comm == nullptr) {
        throw std::invalid_argument(
            "init_gpu_comm: comm must not be nullptr");
    }

    if (id == nullptr) {
        throw std::invalid_argument(
            "init_gpu_comm: id must not be nullptr");
    }

    if (group_size <= 0) {
        throw std::invalid_argument(
            "init_gpu_comm: group_size must be > 0");
    }

    if (local_rank < 0 || local_rank >= group_size) {
        throw std::invalid_argument(
            "init_gpu_comm: local_rank is outside [0, group_size)");
    }

    *comm = nullptr;

    check_collective(
        ncclCommInitRank(comm, group_size, *id, local_rank),
        "ncclCommInitRank");
}

void destroy_gpu_comm(ncclComm_t comm) {
    if (comm != nullptr) {
        check_collective(
            ncclCommDestroy(comm),
            "ncclCommDestroy");
    }
}