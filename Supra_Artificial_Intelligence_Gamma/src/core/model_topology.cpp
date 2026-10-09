#include "core/model_topology.hpp"
#include "gpu/gpu_collectives.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

void check_mpi(int result, const char* what) {
    if (result == MPI_SUCCESS) {
        return;
    }

    char error_text[MPI_MAX_ERROR_STRING] {};
    int length = 0;

    MPI_Error_string(result, error_text, &length);

    throw std::runtime_error(
        std::string(what) +
        " failed: " +
        std::string(error_text, static_cast<std::size_t>(length)));
}

void validate_world(int world_rank, int world_size) {
    if (world_size <= 0) {
        throw std::invalid_argument(
            "ModelTopology: world_size must be greater than zero");
    }

    if (world_rank < 0 || world_rank >= world_size) {
        throw std::invalid_argument(
            "ModelTopology: world_rank must be in [0, world_size)");
    }
}

bool contains_rank(const std::vector<int>& ranks, int rank) {
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

void validate_rank(int rank,
                   int world_size,
                   const std::string& model_id) {
    if (rank < 0 || rank >= world_size) {
        throw std::runtime_error(
            "Invalid MPI rank " + std::to_string(rank) +
            " in topology group '" + model_id +
            "'. Expected rank in [0, " +
            std::to_string(world_size - 1) + "].");
    }

    if (rank == 0) {
        throw std::runtime_error(
            "Topology group '" + model_id +
            "' must not include MPI rank 0: rank 0 is the gateway.");
    }
}

void validate_group_ranks(const ModelGroup& group,
                          int world_size) {
    if (group.model_id.empty()) {
        throw std::runtime_error(
            "Topology contains a model group with an empty model_id.");
    }

    if (group.ranks.empty()) {
        throw std::runtime_error(
            "Topology group '" + group.model_id +
            "' must contain at least one worker rank.");
    }

    std::vector<int> sorted_ranks = group.ranks;
    std::sort(sorted_ranks.begin(), sorted_ranks.end());

    for (std::size_t index = 0; index < sorted_ranks.size(); ++index) {
        validate_rank(sorted_ranks[index], world_size, group.model_id);

        if (index > 0 &&
            sorted_ranks[index] == sorted_ranks[index - 1]) {
            throw std::runtime_error(
                "Topology group '" + group.model_id +
                "' contains duplicate MPI rank " +
                std::to_string(sorted_ranks[index]) + ".");
        }
    }
}

std::string read_json_on_gateway(const std::string& json_path) {
    std::ifstream input(json_path, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "Failed to open topology JSON file: " + json_path);
    }

    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

std::string broadcast_json_from_gateway(int world_rank,
                                        const std::string& json_path) {
    std::string json_text;

    if (world_rank == 0) {
        json_text = read_json_on_gateway(json_path);
    }

    std::uint64_t byte_count =
        static_cast<std::uint64_t>(json_text.size());

    check_mpi(
        MPI_Bcast(&byte_count,
                  1,
                  MPI_UINT64_T,
                  0,
                  MPI_COMM_WORLD),
        "MPI_Bcast(topology JSON size)");

    if (byte_count >
        static_cast<std::uint64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Topology JSON is too large for MPI_Bcast");
    }

    if (world_rank != 0) {
        json_text.resize(static_cast<std::size_t>(byte_count));
    }

    if (byte_count > 0) {
        check_mpi(
            MPI_Bcast(json_text.data(),
                      static_cast<int>(byte_count),
                      MPI_CHAR,
                      0,
                      MPI_COMM_WORLD),
            "MPI_Bcast(topology JSON content)");
    }

    return json_text;
}

std::vector<ModelGroup> parse_topology_json(
    const std::string& json_text,
    const std::string& source_name,
    int world_size) {
    json topology_json;

    try {
        topology_json = json::parse(json_text);
    } catch (const json::exception& error) {
        throw std::runtime_error(
            "Failed to parse topology JSON '" + source_name +
            "': " + error.what());
    }

    if (!topology_json.contains("models") ||
        !topology_json["models"].is_array()) {
        throw std::runtime_error(
            "Invalid topology JSON '" + source_name +
            "': missing array field 'models'.");
    }

    std::unordered_set<std::string> model_ids;
    std::vector<bool> worker_used(
        static_cast<std::size_t>(world_size),
        false);

    std::vector<ModelGroup> groups;

    for (const auto& model_json : topology_json["models"]) {
        if (!model_json.is_object()) {
            throw std::runtime_error(
                "Invalid topology JSON: every entry in 'models' "
                "must be an object.");
        }

        if (!model_json.contains("model_id") ||
            !model_json["model_id"].is_string()) {
            throw std::runtime_error(
                "Invalid topology JSON: model entry is missing "
                "a string field 'model_id'.");
        }

        if (!model_json.contains("ranks") ||
            !model_json["ranks"].is_array()) {
            throw std::runtime_error(
                "Invalid topology JSON: model '" +
                model_json["model_id"].get<std::string>() +
                "' is missing an array field 'ranks'.");
        }

        ModelGroup group;
        group.model_id = model_json["model_id"].get<std::string>();

        if (!model_ids.insert(group.model_id).second) {
            throw std::runtime_error(
                "Duplicate model_id in topology JSON: '" +
                group.model_id + "'.");
        }

        for (const auto& rank_json : model_json["ranks"]) {
            if (!rank_json.is_number_integer()) {
                throw std::runtime_error(
                    "Invalid topology JSON: all ranks in model '" +
                    group.model_id + "' must be integers.");
            }

            group.ranks.push_back(rank_json.get<int>());
        }

        validate_group_ranks(group, world_size);

        for (const int rank : group.ranks) {
            if (worker_used[static_cast<std::size_t>(rank)]) {
                throw std::runtime_error(
                    "MPI worker rank " + std::to_string(rank) +
                    " belongs to more than one model group. "
                    "This topology is unsupported by the current scheduler.");
            }

            worker_used[static_cast<std::size_t>(rank)] = true;
        }

        groups.push_back(std::move(group));
    }

    if (groups.empty()) {
        throw std::runtime_error(
            "Topology JSON '" + source_name +
            "' contains no usable model groups.");
    }

    return groups;
}

}  // namespace

void ModelTopology::init_static(int world_rank, int world_size) {
    finalize();
    groups_.clear();

    validate_world(world_rank, world_size);

    if (world_size < 2) {
        throw std::runtime_error(
            "Static topology requires at least 2 MPI ranks: "
            "rank 0 is the gateway and rank >= 1 is a worker.");
    }

    ModelGroup default_group;
    default_group.model_id = "default";

    for (int rank = 1; rank < world_size; ++rank) {
        default_group.ranks.push_back(rank);
    }

    validate_group_ranks(default_group, world_size);

    create_group_comms(default_group, world_rank, world_size);

    groups_.emplace(
        default_group.model_id,
        std::move(default_group));
}

void ModelTopology::load_from_json(const std::string& json_path,
                                   int world_rank,
                                   int world_size) {
    finalize();
    groups_.clear();

    validate_world(world_rank, world_size);

    if (json_path.empty()) {
        throw std::invalid_argument(
            "ModelTopology::load_from_json: json_path must not be empty");
    }

    if (world_rank == 0) {
        std::cout << "[ModelTopology] Loading topology from JSON: "
                  << json_path
                  << '\n';
    }

    const std::string json_text =
        broadcast_json_from_gateway(world_rank, json_path);

    std::vector<ModelGroup> parsed_groups =
        parse_topology_json(
            json_text,
            json_path,
            world_size);

    // All ranks iterate in the identical JSON array order and therefore call
    // MPI_Comm_split in the same sequence.
    for (auto& group : parsed_groups) {
        create_group_comms(group, world_rank, world_size);

        groups_.emplace(
            group.model_id,
            std::move(group));
    }
}

void ModelTopology::create_group_comms(ModelGroup& group,
                                       int world_rank,
                                       int /* world_size */) {
    const bool belongs_to_group =
        contains_rank(group.ranks, world_rank);

    const int color =
        belongs_to_group ? 1 : MPI_UNDEFINED;

    check_mpi(
        MPI_Comm_split(MPI_COMM_WORLD,
                       color,
                       world_rank,
                       &group.mpi_comm),
        "MPI_Comm_split(model group)");

    if (group.mpi_comm == MPI_COMM_NULL) {
        group.mpi_rank_local = -1;
        group.mpi_size_local = 0;
        group.nccl_comm = nullptr;
        return;
    }

    check_mpi(
        MPI_Comm_rank(group.mpi_comm,
                      &group.mpi_rank_local),
        "MPI_Comm_rank(model group)");

    check_mpi(
        MPI_Comm_size(group.mpi_comm,
                      &group.mpi_size_local),
        "MPI_Comm_size(model group)");

    ncclUniqueId unique_id {};

    if (group.mpi_rank_local == 0) {
        check_collective(
            ncclGetUniqueId(&unique_id),
            "ncclGetUniqueId(model group)");
    }

    check_mpi(
        MPI_Bcast(&unique_id,
                  static_cast<int>(sizeof(unique_id)),
                  MPI_BYTE,
                  0,
                  group.mpi_comm),
        "MPI_Bcast(NCCL/RCCL unique ID)");

    init_gpu_comm(&group.nccl_comm,
                  &unique_id,
                  group.mpi_size_local,
                  group.mpi_rank_local);

    if (group.mpi_rank_local == 0) {
        std::cout << "[ModelTopology] Created MPI + GPU collective "
                  << "communicator for model '"
                  << group.model_id
                  << "' with "
                  << group.mpi_size_local
                  << " rank(s).\n";
    }
}

bool ModelTopology::rank_belongs(const std::string& model_id,
                                 int world_rank) const {
    const auto it = groups_.find(model_id);

    if (it == groups_.end()) {
        return false;
    }

    return contains_rank(it->second.ranks, world_rank);
}

const ModelGroup* ModelTopology::get_group(
    const std::string& model_id) const {
    const auto it = groups_.find(model_id);

    if (it == groups_.end()) {
        return nullptr;
    }

    return &it->second;
}

void ModelTopology::finalize() {
    for (auto& [model_id, group] : groups_) {
        (void) model_id;

        if (group.nccl_comm != nullptr) {
            destroy_gpu_comm(group.nccl_comm);
            group.nccl_comm = nullptr;
        }

        if (group.mpi_comm != MPI_COMM_NULL) {
            check_mpi(
                MPI_Comm_free(&group.mpi_comm),
                "MPI_Comm_free(model group)");

            group.mpi_comm = MPI_COMM_NULL;
        }

        group.mpi_rank_local = -1;
        group.mpi_size_local = 0;
    }
}