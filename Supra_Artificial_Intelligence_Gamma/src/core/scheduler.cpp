#include "scheduler.hpp"

#include <stdexcept>

bool Scheduler::ReqCmp::operator()(const Request& a,
                                   const Request& b) const {
    if (a.priority != b.priority) {
        // std::priority_queue exposes the "largest" element according to
        // Compare. Higher numeric priority must therefore reach top().
        return a.priority < b.priority;
    }

    // For equal priority, dispatch shorter estimated jobs first.
    return a.est_tokens > b.est_tokens;
}

Scheduler::Scheduler(int world_size)
    : world_size_(world_size),
      worker_busy_() {
    if (world_size_ <= 0) {
        throw std::invalid_argument(
            "Scheduler: world_size must be greater than zero");
    }

    worker_busy_.assign(
        static_cast<std::size_t>(world_size_),
        false);

    // Rank 0 is the gateway and never receives worker assignments.
    worker_busy_[0] = true;
}

void Scheduler::enqueue(const Request& req) {
    if (req.est_tokens < 0) {
        throw std::invalid_argument(
            "Scheduler::enqueue: est_tokens must be non-negative");
    }

    queue_.push(req);
}

void Scheduler::set_worker_busy(int worker_rank, bool busy) {
    if (worker_rank <= 0 || worker_rank >= world_size_) {
        throw std::out_of_range(
            "Scheduler::set_worker_busy: worker rank must be in "
            "[1, world_size)");
    }

    worker_busy_[static_cast<std::size_t>(worker_rank)] = busy;
}

std::optional<AssignedRequest> Scheduler::schedule_next() {
    if (queue_.empty()) {
        return std::nullopt;
    }

    int chosen_worker = -1;

    for (int rank = 1; rank < world_size_; ++rank) {
        if (!worker_busy_[static_cast<std::size_t>(rank)]) {
            chosen_worker = rank;
            break;
        }
    }

    if (chosen_worker < 0) {
        // Do not pop: the request remains queued until a worker becomes free.
        return std::nullopt;
    }

    Request request = queue_.top();
    queue_.pop();

    worker_busy_[static_cast<std::size_t>(chosen_worker)] = true;

    return AssignedRequest {
        .req = std::move(request),
        .worker_rank = chosen_worker,
    };
}