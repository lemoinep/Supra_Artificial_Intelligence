#pragma once

#include <string>
#include <vector>
#include <queue>
#include <optional>
#include <cstdint>

// Basic request structure used between gateway and scheduler
struct Request {
    int         id;           // unique request id
    std::string model_id;     // logical model name ("chat", "code", ...)
    std::string prompt;
    int         priority;     // higher = more important
    int         est_tokens;   // estimated output length (for SJF-like scheduling)
};

// Assignment decided by the scheduler
struct AssignedRequest {
    Request req;
    int     worker_rank;      // MPI rank of chosen worker
};

// Simple multi-model scheduler
class Scheduler {
public:
    explicit Scheduler(int world_size);

    // Enqueue a new request
    void enqueue(const Request& req);

    // Try to schedule the next request onto some worker.
    // Returns std::nullopt if no request can be scheduled now.
    std::optional<AssignedRequest> schedule_next();

    // Mark a worker as busy or free
    void set_worker_busy(int worker_rank, bool busy);

private:
    // Priority comparator:
    //  - higher priority first
    //  - for equal priority, smaller est_tokens first (shortest-job-first)
    struct ReqCmp {
        bool operator()(const Request& a, const Request& b) const;
    };

    int world_size_;
    std::priority_queue<Request, std::vector<Request>, ReqCmp> queue_;
    std::vector<bool> worker_busy_; // worker_busy_[rank] = true if worker busy
};