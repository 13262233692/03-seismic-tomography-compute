#pragma once

#include "snapshot_buffer.h"
#include <cstdint>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <fstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace seismic {

constexpr int64_t DEFAULT_MMAP_MAX_BYTES = 4LL * 1024LL * 1024LL * 1024LL;
constexpr int64_t MMAP_FILE_MAGIC = 0x52544D534E4150ULL;

#pragma pack(push, 1)
struct MmapFileHeader {
    int64_t magic;
    int64_t header_size;
    int64_t file_size_bytes;
    int64_t frame_count;
    int64_t write_offset;
    int64_t ring_start;
    int64_t ring_capacity;
    int64_t reserved[8];
};
#pragma pack(pop)

class MmapSnapshotWriter {
public:
    MmapSnapshotWriter();
    ~MmapSnapshotWriter();

    bool open(const std::string& filename, int64_t max_bytes = DEFAULT_MMAP_MAX_BYTES);
    void close();
    bool is_open() const { return is_open_; }

    bool write_frame(const uint8_t* frame_data, int64_t frame_size);

    int64_t frame_count() const { return header()->frame_count; }
    int64_t write_offset() const { return header()->write_offset; }
    int64_t used_bytes() const { return write_offset(); }
    int64_t capacity_bytes() const { return max_bytes_; }

    void set_async_mode(bool async) { async_mode_ = async; }

    void start_async_drain(LockFreeSnapshotBuffer& buffer);
    void stop_async_drain();
    bool async_drain_running() const { return drain_running_.load(std::memory_order_acquire); }

    struct Stats {
        int64_t frames_written;
        int64_t frames_dropped_buffer_full;
        int64_t total_bytes_written;
        int64_t mmap_cycles;
    };

    Stats stats() const;

    std::string filename() const { return filename_; }

    static bool read_frame(const std::string& filename,
                            int64_t frame_index,
                            std::vector<uint8_t>& out_data);

    static int64_t count_frames(const std::string& filename);

    MmapFileHeader* header() {
        return reinterpret_cast<MmapFileHeader*>(mmap_base_);
    }

    const MmapFileHeader* header() const {
        return reinterpret_cast<const MmapFileHeader*>(mmap_base_);
    }

private:
    std::string filename_;
    bool is_open_;
    bool async_mode_;

    int64_t max_bytes_;
    void* mmap_base_;
    size_t mmap_size_;

    uint8_t* frame_region_;
    int64_t frame_region_size_;

    std::atomic<bool> drain_running_;
    std::thread drain_thread_;

    std::atomic<int64_t> frames_written_;
    std::atomic<int64_t> frames_dropped_;
    std::atomic<int64_t> total_bytes_written_;
    std::atomic<int64_t> mmap_cycles_;

#ifdef _WIN32
    HANDLE file_handle_;
    HANDLE map_handle_;
#else
    int fd_;
#endif

    void drain_thread_fn(LockFreeSnapshotBuffer* buffer);
    bool create_or_remap(const std::string& filename, int64_t bytes);
    bool grow_mapping();
    void reset_write_pointer();

    MmapSnapshotWriter(const MmapSnapshotWriter&) = delete;
    MmapSnapshotWriter& operator=(const MmapSnapshotWriter&) = delete;
};

}
