#pragma once

#include "core/scheduler.hpp"

#include <atomic>
#include <condition_variable>
#include <optional>
#include <mutex>

// -----------------------------------------------------------------------------
// Thread-safe wrapper around Scheduler.
//
// Used by the gateway rank to synchronize:
//   - HTTP request handlers that enqueue work,
//   - the dispatcher thread that assigns work to MPI workers,
//   - shutdown signaling.
//
// Scheduler itself remains single-threaded and backend-agnostic.
// -----------------------------------------------------------------------------
class ThreadSafeScheduler {
public:
    explicit ThreadSafeScheduler(int world_size);

    ThreadSafeScheduler(const ThreadSafeScheduler&) = delete;
    ThreadSafeScheduler& operator=(const ThreadSafeScheduler&) = delete;

    // Add a request and wake one dispatcher waiting for work.
    void enqueue(const Request& request);

    // Non-blocking attempt to assign one queued request to a free worker.
    // Returns std::nullopt when the queue is empty or every worker is busy.
    std::optional<AssignedRequest> try_schedule_next();

    // Wait until:
    //   - a request can be assigned to a free worker, or
    //   - shutdown becomes true.
    //
    // Returns std::nullopt after shutdown.
    std::optional<AssignedRequest> wait_and_schedule_next(
        const std::atomic<bool>& shutdown);

    // Update the availability state of an MPI worker rank.
    // A worker becoming free wakes a potential dispatcher.
    void set_worker_busy(int worker_rank, bool busy);

    // Wake every thread currently blocked in wait_and_schedule_next().
    //
    // Call this immediately after setting shutdown = true, otherwise a thread
    // already sleeping on cv_ may not wake until another request arrives.
    void notify_shutdown();

private:
    Scheduler scheduler_;
    std::mutex mutex_;
    std::condition_variable cv_;
};