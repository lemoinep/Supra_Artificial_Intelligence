#include "scheduler_threadsafe.hpp"

ThreadSafeScheduler::ThreadSafeScheduler(int world_size)
    : sched_(world_size) {
}

void ThreadSafeScheduler::enqueue(const Request& req) {
    {
        std::lock_guard<std::mutex> lock(m_);
        sched_.enqueue(req);
    }

    cv_.notify_one();
}

std::optional<AssignedRequest> ThreadSafeScheduler::try_schedule_next() {
    std::lock_guard<std::mutex> lock(m_);
    return sched_.schedule_next();
}

std::optional<AssignedRequest>
ThreadSafeScheduler::wait_and_schedule_next(bool& shutdown) {
    std::unique_lock<std::mutex> lock(m_);

    cv_.wait(lock, [&] {
        return shutdown || sched_.can_schedule();
    });

    if (shutdown) {
        return std::nullopt;
    }

    // The mutex is still held, so no other ThreadSafeScheduler operation can
    // change queue_ or worker_busy_ between can_schedule() and this call.
    return sched_.schedule_next();
}

void ThreadSafeScheduler::set_worker_busy(int worker_rank, bool busy) {
    {
        std::lock_guard<std::mutex> lock(m_);
        sched_.set_worker_busy(worker_rank, busy);
    }

    // A newly free worker may make a queued request schedulable.
    if (!busy) {
        cv_.notify_one();
    }
}