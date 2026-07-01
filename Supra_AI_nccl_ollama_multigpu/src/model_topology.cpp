#include "model_topology.hpp"
#include "nccl_utils.hpp"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static bool contains_rank(const std::vector<int>& v, int r) {
    return std::find(v.begin(), v.end(), r) != v.end();
}

void ModelTopology::init_static(int world_rank, int world_size) {
    finalize();
    groups_.clear();

    ModelGroup default_group;
    default_group.model_id = "default";
    for (int r = 1; r < world_size; ++r) {
        default_group.ranks.push_back(r);
    }

    create_group_comms(default_group, world_rank, world_size);
    groups_.emplace(default_group.model_id, std::move(default_group));
}

void ModelTopology::load_from_json(const std::string& json_path,
                                   int world_rank,
                                   int world_size)
{
    finalize();
    groups_.clear();

    if (world_rank == 0) {
        std::cout << "[ModelTopology] Loading topology from JSON: "
                  << json_path << std::endl;
    }

    std::ifstream in(json_path);
    if (!in) {
        throw std::runtime_error("Failed to open model_topology JSON: " + json_path);
    }

    json j;
    in >> j;

    if (!j.contains("models") || !j["models"].is_array()) {
        throw std::runtime_error("Invalid JSON: 'models' array missing");
    }

    for (const auto& jm : j["models"]) {
        if (!jm.contains("model_id") || !jm.contains("ranks")) {
            continue;
        }
        ModelGroup group;
        group.model_id = jm["model_id"].get<std::string>();

        for (const auto& r : jm["ranks"]) {
            group.ranks.push_back(r.get<int>());
        }

        create_group_comms(group, world_rank, world_size);
        groups_.emplace(group.model_id, std::move(group));
    }
}

void ModelTopology::create_group_comms(ModelGroup& group,
                                       int world_rank,
                                       int world_size)
{
    int color = MPI_UNDEFINED;
    if (contains_rank(group.ranks, world_rank)) {
        color = 1; // simple: 1 pour tous les groupes; à raffiner si besoin
    }

    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &new_comm);
    group.mpi_comm = new_comm;

    if (new_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(new_comm, &group.mpi_rank_local);
        MPI_Comm_size(new_comm, &group.mpi_size_local);

        ncclUniqueId id;
        if (group.mpi_rank_local == 0) {
            checkNccl(ncclGetUniqueId(&id), "ncclGetUniqueId(model)");
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, new_comm);

        ncclComm_t model_comm = nullptr;
        checkNccl(
            ncclCommInitRank(&model_comm,
                             group.mpi_size_local,
                             id,
                             group.mpi_rank_local),
            "ncclCommInitRank(model_group)");

        group.nccl_comm = model_comm;

        if (world_rank == group.ranks.front()) {
            std::cout << "[ModelTopology] Created MPI+NCCL comm for model '"
                      << group.model_id << "' with "
                      << group.mpi_size_local << " ranks." << std::endl;
        }
    } else {
        group.mpi_rank_local = -1;
        group.mpi_size_local = 0;
        group.nccl_comm      = nullptr;
    }
}

bool ModelTopology::rank_belongs(const std::string& model_id,
                                 int world_rank) const
{
    auto it = groups_.find(model_id);
    if (it == groups_.end()) return false;
    return contains_rank(it->second.ranks, world_rank);
}

const ModelGroup* ModelTopology::get_group(const std::string& model_id) const {
    auto it = groups_.find(model_id);
    if (it == groups_.end()) return nullptr;
    return &it->second;
}

void ModelTopology::finalize() {
    for (auto& kv : groups_) {
        ModelGroup& g = kv.second;
        if (g.nccl_comm != nullptr) {
            ncclCommDestroy(g.nccl_comm);
            g.nccl_comm = nullptr;
        }
        if (g.mpi_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&g.mpi_comm);
            g.mpi_comm = MPI_COMM_NULL;
        }
        g.mpi_rank_local = -1;
        g.mpi_size_local = 0;
    }
}