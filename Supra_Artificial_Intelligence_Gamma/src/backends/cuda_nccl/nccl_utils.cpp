#include "nccl_utils.hpp"

void init_nccl_comm(ncclComm_t* comm,
                    const ncclUniqueId* id,
                    int world_size,
                    int world_rank)
{
    checkNccl(
        ncclCommInitRank(comm, world_size, *id, world_rank),
        "ncclCommInitRank");
}

void destroy_nccl_comm(ncclComm_t comm) {
    if (comm != nullptr) {
        ncclCommDestroy(comm);
    }
}