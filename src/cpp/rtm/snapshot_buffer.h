#pragma once

#include "rtm_engine.h"
#include <cstdint>
#include <vector>
#include <atomic>
#include <thread>
#include <cstring>
#include <memory>
#include <array>
#include <chrono>

namespace seismic {

constexpr size_t DEFAULT_RING_BUFFER_SIZE = 1024;
constexpr size_t SLICE_HEADER_SIZE = 128;

#pragma pack(push, 1)
struct SliceFrameHeader {
    uint64_t magic;
    int64_t  timestep;
    int32_t  axis;
    int64_t  index;
    int64_t  dim0;
    int64_t  dim1;
    int64_t  frame_size_bytes;
    int64_t  forward_offset;
    int64_t  adjoint_offset;
    int64_t  ic_offset;
    int64_t  forward_bytes;
    int64_t  adjoint_bytes;
    int64_t  ic_bytes;
    double   capture_timestamp;
    uint64_t sequence_id;
    uint32_t checksum;
    uint32_t reserved;
};
#pragma pack(pop)

struct RingBufferSlot {
    std::atomic<int> status;
    std::vector<uint8_t> payload;
    std::atomic<uint64_t> sequence;
};

enum SlotStatus : int {
    SLOT_EMPTY = 0,
    SLOT_WRITING = 1,
    SLOT_FULL = 2,
    SLOT_READING = 3,
    SLOT_OVERFLOW = 4
};

class LockFreeSnapshotBuffer {
public:
    explicit LockFreeSnapshotBuffer(size_t capacity = DEFAULT_RING_BUFFER_SIZE);

    ~LockFreeSnapshotBuffer();

    bool try_enqueue(const RTMEngine::CapturedSlice& slice);

    bool try_dequeue(RTMEngine::CapturedSlice& out_slice);

    size_t size_approx() const {
        return (write_index_.load(std::memory_order_acquire) -
                read_index_.load(std::memory_order_acquire));
    }

    size_t capacity() const { return capacity_; }

    bool is_empty() const {
        return write_index_.load(std::memory_order_acquire) ==
               read_index_.load(std::memory_order_acquire);
    }

    bool is_full() const {
        return size_approx() >= capacity_;
    }

    uint64_t total_enqueued() const { return total_enqueued_.load(std::memory_order_acquire); }
    uint64_t total_dropped() const { return total_dropped_.load(std::memory_order_acquire); }
    uint64_t total_read() const { return total_read_.load(std::memory_order_acquire); }

    void reset();

    void serialize_slice(const RTMEngine::CapturedSlice& slice,
                          std::vector<uint8_t>& buffer);

    bool deserialize_slice(const uint8_t* buffer, size_t buffer_size,
                            RTMEngine::CapturedSlice& out_slice);

private:
    size_t capacity_;
    std::vector<RingBufferSlot> slots_;
    std::atomic<size_t> write_index_;
    std::atomic<size_t> read_index_;
    std::atomic<uint64_t> sequence_counter_;
    std::atomic<uint64_t> total_enqueued_;
    std::atomic<uint64_t> total_dropped_;
    std::atomic<uint64_t> total_read_;

    size_t mask_;

    static uint32_t compute_checksum(const uint8_t* data, size_t size);
};

}
