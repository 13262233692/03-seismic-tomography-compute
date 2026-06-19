#include "snapshot_buffer.h"
#include <algorithm>
#include <cassert>

namespace seismic {

LockFreeSnapshotBuffer::LockFreeSnapshotBuffer(size_t capacity) {
    capacity_ = 1;
    while (capacity_ < capacity) capacity_ <<= 1;
    mask_ = capacity_ - 1;

    slots_.resize(capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
        slots_[i].status.store(SLOT_EMPTY, std::memory_order_release);
        slots_[i].sequence.store(0, std::memory_order_release);
    }

    write_index_.store(0, std::memory_order_release);
    read_index_.store(0, std::memory_order_release);
    sequence_counter_.store(0, std::memory_order_release);
    total_enqueued_.store(0, std::memory_order_release);
    total_dropped_.store(0, std::memory_order_release);
    total_read_.store(0, std::memory_order_release);
}

LockFreeSnapshotBuffer::~LockFreeSnapshotBuffer() = default;

bool LockFreeSnapshotBuffer::try_enqueue(const RTMEngine::CapturedSlice& slice) {
    size_t write_idx = write_index_.load(std::memory_order_acquire);
    size_t slot_idx = write_idx & mask_;

    int expected = SLOT_EMPTY;
    if (!slots_[slot_idx].status.compare_exchange_strong(
            expected, SLOT_WRITING,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        if (expected == SLOT_FULL || expected == SLOT_READING) {
            total_dropped_.fetch_add(1, std::memory_order_release);
            slots_[slot_idx].status.store(SLOT_OVERFLOW, std::memory_order_release);
            slots_[slot_idx].status.store(SLOT_EMPTY, std::memory_order_release);
            return false;
        }
    }

    serialize_slice(slice, slots_[slot_idx].payload);
    uint64_t seq = sequence_counter_.fetch_add(1, std::memory_order_acq_rel) + 1;
    slots_[slot_idx].sequence.store(seq, std::memory_order_release);

    slots_[slot_idx].status.store(SLOT_FULL, std::memory_order_release);

    size_t new_write_idx = write_idx + 1;
    write_index_.store(new_write_idx, std::memory_order_release);
    total_enqueued_.fetch_add(1, std::memory_order_release);
    return true;
}

bool LockFreeSnapshotBuffer::try_dequeue(RTMEngine::CapturedSlice& out_slice) {
    size_t read_idx = read_index_.load(std::memory_order_acquire);
    size_t slot_idx = read_idx & mask_;

    int expected = SLOT_FULL;
    if (!slots_[slot_idx].status.compare_exchange_strong(
            expected, SLOT_READING,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return false;
    }

    bool success = deserialize_slice(
        slots_[slot_idx].payload.data(),
        slots_[slot_idx].payload.size(),
        out_slice
    );

    slots_[slot_idx].payload.clear();
    slots_[slot_idx].status.store(SLOT_EMPTY, std::memory_order_release);

    size_t new_read_idx = read_idx + 1;
    read_index_.store(new_read_idx, std::memory_order_release);

    if (success) {
        total_read_.fetch_add(1, std::memory_order_release);
    }
    return success;
}

void LockFreeSnapshotBuffer::reset() {
    for (size_t i = 0; i < capacity_; ++i) {
        slots_[i].payload.clear();
        slots_[i].status.store(SLOT_EMPTY, std::memory_order_release);
        slots_[i].sequence.store(0, std::memory_order_release);
    }
    write_index_.store(0, std::memory_order_release);
    read_index_.store(0, std::memory_order_release);
    sequence_counter_.store(0, std::memory_order_release);
}

void LockFreeSnapshotBuffer::serialize_slice(
    const RTMEngine::CapturedSlice& slice,
    std::vector<uint8_t>& buffer) {

    SliceFrameHeader hdr{};
    hdr.magic = 0x534C494345303031ULL;
    hdr.timestep = slice.timestep;
    hdr.axis = static_cast<int32_t>(slice.axis);
    hdr.index = slice.index;
    hdr.dim0 = slice.dim0;
    hdr.dim1 = slice.dim1;

    hdr.forward_bytes = slice.forward_slice.size() * sizeof(double);
    hdr.adjoint_bytes = slice.adjoint_slice.size() * sizeof(double);
    hdr.ic_bytes = slice.ic_slice.size() * sizeof(double);

    hdr.forward_offset = SLICE_HEADER_SIZE;
    hdr.adjoint_offset = hdr.forward_offset + hdr.forward_bytes;
    hdr.ic_offset = hdr.adjoint_offset + hdr.adjoint_bytes;

    hdr.frame_size_bytes = SLICE_HEADER_SIZE + hdr.forward_bytes + hdr.adjoint_bytes + hdr.ic_bytes;
    hdr.capture_timestamp = slice.capture_timestamp;
    hdr.sequence_id = 0;
    hdr.checksum = 0;
    hdr.reserved = 0;

    buffer.clear();
    buffer.resize(static_cast<size_t>(hdr.frame_size_bytes));

    std::memcpy(buffer.data(), &hdr, SLICE_HEADER_SIZE);

    if (hdr.forward_bytes > 0) {
        std::memcpy(buffer.data() + hdr.forward_offset,
                    slice.forward_slice.data(),
                    static_cast<size_t>(hdr.forward_bytes));
    }
    if (hdr.adjoint_bytes > 0) {
        std::memcpy(buffer.data() + hdr.adjoint_offset,
                    slice.adjoint_slice.data(),
                    static_cast<size_t>(hdr.adjoint_bytes));
    }
    if (hdr.ic_bytes > 0) {
        std::memcpy(buffer.data() + hdr.ic_offset,
                    slice.ic_slice.data(),
                    static_cast<size_t>(hdr.ic_bytes));
    }

    hdr.checksum = compute_checksum(
        buffer.data() + SLICE_HEADER_SIZE,
        static_cast<size_t>(hdr.frame_size_bytes - SLICE_HEADER_SIZE)
    );
    std::memcpy(buffer.data(), &hdr, SLICE_HEADER_SIZE);
}

bool LockFreeSnapshotBuffer::deserialize_slice(
    const uint8_t* buffer, size_t buffer_size,
    RTMEngine::CapturedSlice& out_slice) {

    if (buffer_size < SLICE_HEADER_SIZE) return false;

    SliceFrameHeader hdr{};
    std::memcpy(&hdr, buffer, SLICE_HEADER_SIZE);

    if (hdr.magic != 0x534C494345303031ULL) return false;
    if (static_cast<size_t>(hdr.frame_size_bytes) > buffer_size) return false;

    uint32_t expected_csum = hdr.checksum;
    hdr.checksum = 0;
    uint32_t actual_csum = compute_checksum(
        buffer + SLICE_HEADER_SIZE,
        static_cast<size_t>(hdr.frame_size_bytes - SLICE_HEADER_SIZE)
    );
    if (expected_csum != actual_csum) return false;

    out_slice.timestep = hdr.timestep;
    out_slice.axis = static_cast<SliceAxis>(hdr.axis);
    out_slice.index = hdr.index;
    out_slice.dim0 = hdr.dim0;
    out_slice.dim1 = hdr.dim1;
    out_slice.capture_timestamp = hdr.capture_timestamp;

    size_t forward_n = static_cast<size_t>(hdr.forward_bytes / sizeof(double));
    size_t adjoint_n = static_cast<size_t>(hdr.adjoint_bytes / sizeof(double));
    size_t ic_n = static_cast<size_t>(hdr.ic_bytes / sizeof(double));

    out_slice.forward_slice.resize(forward_n);
    out_slice.adjoint_slice.resize(adjoint_n);
    out_slice.ic_slice.resize(ic_n);

    if (hdr.forward_bytes > 0) {
        std::memcpy(out_slice.forward_slice.data(),
                    buffer + hdr.forward_offset,
                    static_cast<size_t>(hdr.forward_bytes));
    }
    if (hdr.adjoint_bytes > 0) {
        std::memcpy(out_slice.adjoint_slice.data(),
                    buffer + hdr.adjoint_offset,
                    static_cast<size_t>(hdr.adjoint_bytes));
    }
    if (hdr.ic_bytes > 0) {
        std::memcpy(out_slice.ic_slice.data(),
                    buffer + hdr.ic_offset,
                    static_cast<size_t>(hdr.ic_bytes));
    }

    return true;
}

uint32_t LockFreeSnapshotBuffer::compute_checksum(const uint8_t* data, size_t size) {
    uint32_t sum = 0;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(data);
    size_t n32 = size / sizeof(uint32_t);
    size_t n_rest = size % sizeof(uint32_t);

    for (size_t i = 0; i < n32; ++i) {
        sum ^= p[i];
        sum = (sum << 13) | (sum >> 19);
    }

    if (n_rest > 0) {
        uint32_t rest = 0;
        const uint8_t* r = data + n32 * sizeof(uint32_t);
        for (size_t i = 0; i < n_rest; ++i) {
            rest |= (uint32_t(r[i]) << (8 * i));
        }
        sum ^= rest;
    }

    return sum;
}

}
