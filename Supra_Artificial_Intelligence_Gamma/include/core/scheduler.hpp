#pragma once

#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Gateway request waiting to be scheduled on a worker.
//
// This structure is backend-agnostic: it contains no CUDA, HIP, NCCL or RCCL
// detail. It describes only the work requested by an HTTP/CLI client.
// -----------------------------------------------------------------------------
struct Request {
    int id = -1;                 // Unique request identifier.
    std::string model_id;        // Logical model name: "default", "chat", "code".
    std::string prompt;          // Input prompt after gateway formatting.
    int priority = 0;            // Higher value means higher priority.
    int est_tokens = 0;          // Estimated generation length for SJF ordering.
};

// -----------------------------------------------------------------------------
// Result of one scheduling decision.
// -----------------------------------------------------------------------------
struct AssignedRequest {
    Request req;
    int worker_rank = -1;        // MPI world rank selected by Scheduler.
};

// -----------------------------------------------------------------------------
// Simple priority + shortest-job-first scheduler.
//
// Scheduling policy:
//   1. Higher request priority first.
//   2. For equal priority, lowest estimated token count first.
//   3. Assign to the first available worker rank.
//
// Rank 0 is reserved for the gateway. Worker ranks normally start at 1.
//
// Note:
// This initial scheduler does not yet apply model_id -> worker-group placement.
// That will be added after ModelTopology is connected to the scheduling layer.
// -----------------------------------------------------------------------------
class Scheduler {
public:
    explicit Scheduler(int world_size);

    // Enqueue a request. Thread safety is provided by ThreadSafeScheduler.
    void enqueue(const Request& request);

    // Attempt one scheduling decision without blocking.
    // Returns std::nullopt when there is no queued work or no available worker.
    std::optional<AssignedRequest> schedule_next();

    // Mark a worker MPI rank as busy or available.
    void set_worker_busy(int worker_rank, bool busy);

private:
    // Comparator for std::priority_queue.
    //
    // std::priority_queue retains the element considered "largest" according
    // to this ordering at top(). This comparator makes higher priorities win;
    // for equal priorities, lower est_tokens wins.
    struct RequestComparator {
        bool operator()(const Request& lhs,
                        const Request& rhs) const;
    };

    int world_size_ = 1;

    std::priority_queue<Request,
                        std::vector<Request>,
                        RequestComparator>
        queue_;

    // worker_busy_[rank] == true means that MPI rank cannot receive work.
    // Index 0 is permanently reserved for the gateway.
    std::vector<bool> worker_busy_;
};