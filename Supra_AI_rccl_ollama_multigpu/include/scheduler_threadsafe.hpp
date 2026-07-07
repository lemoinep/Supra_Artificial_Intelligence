#pragma once

#include "scheduler.hpp"
#include <mutex>
#include <condition_variable>

// Thread-safe wrapper around Scheduler for the gateway (HTTP + dispatcher).
class ThreadSafeScheduler {
public:
    explicit ThreadSafeScheduler(int world_size);

    // Enqueue request, wake dispatcher if needed.
    void enqueue(const Request& req);

    // Non-blocking attempt to schedule.
    std::optional<AssignedRequest> try_schedule_next();

    // Blocking wait until a request can be scheduled or shutdown flag is set.
    std::optional<AssignedRequest> wait_and_schedule_next(bool& shutdown);

    void set_worker_busy(int worker_rank, bool busy);

private:
    Scheduler sched_;
    std::mutex m_;
    std::condition_variable cv_;
};