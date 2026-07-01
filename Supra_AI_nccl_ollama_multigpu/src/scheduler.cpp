#include "scheduler.hpp"

bool Scheduler::ReqCmp::operator()(const Request& a, const Request& b) const {
    if (a.priority != b.priority) {
        return a.priority < b.priority; // higher priority first
    }
    return a.est_tokens > b.est_tokens; // shorter jobs first
}

Scheduler::Scheduler(int world_size)
    : world_size_(world_size),
      worker_busy_(world_size, false)
{
    if (world_size_ > 0) {
        worker_busy_[0] = true; // rank 0 = gateway (not a worker)
    }
}

void Scheduler::enqueue(const Request& req) {
    queue_.push(req);
}

void Scheduler::set_worker_busy(int worker_rank, bool busy) {
    if (worker_rank >= 0 && worker_rank < (int)worker_busy_.size()) {
        worker_busy_[worker_rank] = busy;
    }
}

std::optional<AssignedRequest> Scheduler::schedule_next() {
    if (queue_.empty()) {
        return std::nullopt;
    }

    int chosen_worker = -1;
    for (int r = 1; r < world_size_; ++r) {
        if (!worker_busy_[r]) {
            chosen_worker = r;
            break;
        }
    }
    if (chosen_worker < 0) {
        return std::nullopt;
    }

    Request req = queue_.top();
    queue_.pop();

    worker_busy_[chosen_worker] = true;

    AssignedRequest ar;
    ar.req         = req;
    ar.worker_rank = chosen_worker;
    return ar;
}