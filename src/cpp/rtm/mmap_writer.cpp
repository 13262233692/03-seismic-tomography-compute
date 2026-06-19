#include "mmap_writer.h"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#else
#include <cerrno>
#endif

namespace seismic {

MmapSnapshotWriter::MmapSnapshotWriter()
    : is_open_(false), async_mode_(false),
      max_bytes_(0), mmap_base_(nullptr), mmap_size_(0),
      frame_region_(nullptr), frame_region_size_(0),
      frames_written_(0), frames_dropped_(0),
      total_bytes_written_(0), mmap_cycles_(0)
#ifdef _WIN32
      , file_handle_(INVALID_HANDLE_VALUE), map_handle_(NULL)
#else
      , fd_(-1)
#endif
{
    drain_running_.store(false, std::memory_order_release);
}

MmapSnapshotWriter::~MmapSnapshotWriter() {
    stop_async_drain();
    close();
}

bool MmapSnapshotWriter::open(const std::string& filename, int64_t max_bytes) {
    close();
    filename_ = filename;
    max_bytes_ = max_bytes;
    return create_or_remap(filename, max_bytes);
}

bool MmapSnapshotWriter::create_or_remap(const std::string& filename, int64_t bytes) {
#ifdef _WIN32
    file_handle_ = CreateFileA(
        filename.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        NULL
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER li;
    li.QuadPart = bytes;
    map_handle_ = CreateFileMappingA(
        file_handle_, NULL, PAGE_READWRITE,
        li.HighPart, li.LowPart, NULL
    );
    if (map_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    mmap_base_ = MapViewOfFile(
        map_handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0
    );
    if (mmap_base_ == nullptr) {
        CloseHandle(map_handle_);
        CloseHandle(file_handle_);
        map_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    mmap_size_ = static_cast<size_t>(bytes);
#else
    fd_ = ::open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) return false;

    if (::ftruncate(fd_, bytes) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    mmap_base_ = ::mmap(nullptr, static_cast<size_t>(bytes),
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        mmap_base_ = nullptr;
        return false;
    }
    mmap_size_ = static_cast<size_t>(bytes);
#endif

    MmapFileHeader hdr{};
    hdr.magic = MMAP_FILE_MAGIC;
    hdr.header_size = sizeof(MmapFileHeader);
    hdr.file_size_bytes = bytes;
    hdr.frame_count = 0;
    hdr.write_offset = sizeof(MmapFileHeader);
    hdr.ring_start = 0;
    hdr.ring_capacity = 0;

    std::memcpy(mmap_base_, &hdr, sizeof(hdr));

    frame_region_ = reinterpret_cast<uint8_t*>(mmap_base_) + sizeof(MmapFileHeader);
    frame_region_size_ = bytes - sizeof(MmapFileHeader);

    is_open_ = true;
    mmap_cycles_.fetch_add(1, std::memory_order_release);
    return true;
}

void MmapSnapshotWriter::close() {
    if (!is_open_) return;

    if (mmap_base_) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_base_);
        CloseHandle(map_handle_);
        CloseHandle(file_handle_);
        map_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
#else
        ::msync(mmap_base_, mmap_size_, MS_SYNC);
        ::munmap(mmap_base_, mmap_size_);
        ::close(fd_);
        fd_ = -1;
#endif
        mmap_base_ = nullptr;
        mmap_size_ = 0;
    }

    is_open_ = false;
}

bool MmapSnapshotWriter::write_frame(const uint8_t* frame_data, int64_t frame_size) {
    if (!is_open_ || frame_data == nullptr || frame_size <= 0) return false;

    int64_t frame_with_size = frame_size + sizeof(int64_t);

    MmapFileHeader* hdr = header();
    int64_t current_offset = __atomic_load_n(&hdr->write_offset, __ATOMIC_ACQUIRE);

    if (current_offset + frame_with_size > max_bytes_) {
        reset_write_pointer();
        current_offset = sizeof(MmapFileHeader);
    }

    uint8_t* dst = reinterpret_cast<uint8_t*>(mmap_base_) + current_offset;
    std::memcpy(dst, &frame_size, sizeof(int64_t));
    std::memcpy(dst + sizeof(int64_t), frame_data, static_cast<size_t>(frame_size));

    int64_t new_offset = current_offset + frame_with_size;
    __atomic_store_n(&hdr->write_offset, new_offset, __ATOMIC_RELEASE);
    int64_t new_count = __atomic_add_fetch(&hdr->frame_count, 1, __ATOMIC_ACQ_REL);
    (void)new_count;

    frames_written_.fetch_add(1, std::memory_order_release);
    total_bytes_written_.fetch_add(frame_with_size, std::memory_order_release);

    return true;
}

void MmapSnapshotWriter::reset_write_pointer() {
    MmapFileHeader* hdr = header();
    __atomic_store_n(&hdr->write_offset, sizeof(MmapFileHeader), __ATOMIC_RELEASE);
    __atomic_fetch_add(&hdr->ring_start, 1, __ATOMIC_ACQ_REL);
    mmap_cycles_.fetch_add(1, std::memory_order_release);
}

void MmapSnapshotWriter::start_async_drain(LockFreeSnapshotBuffer& buffer) {
    if (drain_running_.load(std::memory_order_acquire)) return;
    drain_running_.store(true, std::memory_order_release);
    drain_thread_ = std::thread(&MmapSnapshotWriter::drain_thread_fn, this, &buffer);
}

void MmapSnapshotWriter::stop_async_drain() {
    if (!drain_running_.load(std::memory_order_acquire)) return;
    drain_running_.store(false, std::memory_order_release);
    if (drain_thread_.joinable()) {
        drain_thread_.join();
    }
}

void MmapSnapshotWriter::drain_thread_fn(LockFreeSnapshotBuffer* buffer) {
    LockFreeSnapshotBuffer& buf = *buffer;
    LockFreeSnapshotBuffer local_deserialize_buf(64);

    std::vector<uint8_t> serialized_frame;
    RTMEngine::CapturedSlice tmp_slice;

    while (drain_running_.load(std::memory_order_acquire)) {
        bool had_work = false;

        while (buf.try_dequeue(tmp_slice)) {
            had_work = true;
            local_deserialize_buf.serialize_slice(tmp_slice, serialized_frame);
            if (!write_frame(serialized_frame.data(),
                             static_cast<int64_t>(serialized_frame.size()))) {
                frames_dropped_.fetch_add(1, std::memory_order_release);
            }
        }

        if (!had_work) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    while (buf.try_dequeue(tmp_slice)) {
        local_deserialize_buf.serialize_slice(tmp_slice, serialized_frame);
        write_frame(serialized_frame.data(),
                    static_cast<int64_t>(serialized_frame.size()));
    }
}

MmapSnapshotWriter::Stats MmapSnapshotWriter::stats() const {
    Stats s{};
    s.frames_written = frames_written_.load(std::memory_order_acquire);
    s.frames_dropped_buffer_full = frames_dropped_.load(std::memory_order_acquire);
    s.total_bytes_written = total_bytes_written_.load(std::memory_order_acquire);
    s.mmap_cycles = mmap_cycles_.load(std::memory_order_acquire);
    return s;
}

bool MmapSnapshotWriter::read_frame(const std::string& filename,
                                     int64_t frame_index,
                                     std::vector<uint8_t>& out_data) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER fs;
    GetFileSizeEx(fh, &fs);
    int64_t file_size = fs.QuadPart;

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mh == NULL) { CloseHandle(fh); return false; }

    void* base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) { CloseHandle(mh); CloseHandle(fh); return false; }
#else
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st;
    if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
    int64_t file_size = st.st_size;

    void* base = ::mmap(nullptr, static_cast<size_t>(file_size),
                        PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { ::close(fd); return false; }
#endif

    const MmapFileHeader* hdr = reinterpret_cast<const MmapFileHeader*>(base);
    if (hdr->magic != MMAP_FILE_MAGIC) {
#ifdef _WIN32
        UnmapViewOfFile(base); CloseHandle(mh); CloseHandle(fh);
#else
        ::munmap(base, static_cast<size_t>(file_size)); ::close(fd);
#endif
        return false;
    }

    int64_t offset = sizeof(MmapFileHeader);
    int64_t end = hdr->write_offset;
    int64_t current_idx = 0;
    bool found = false;

    while (offset + sizeof(int64_t) <= end) {
        int64_t frame_size = *reinterpret_cast<const int64_t*>(
            reinterpret_cast<const uint8_t*>(base) + offset);
        offset += sizeof(int64_t);

        if (current_idx == frame_index) {
            out_data.resize(static_cast<size_t>(frame_size));
            std::memcpy(out_data.data(),
                        reinterpret_cast<const uint8_t*>(base) + offset,
                        static_cast<size_t>(frame_size));
            found = true;
            break;
        }

        offset += frame_size;
        ++current_idx;
    }

#ifdef _WIN32
    UnmapViewOfFile(base); CloseHandle(mh); CloseHandle(fh);
#else
    ::munmap(base, static_cast<size_t>(file_size)); ::close(fd);
#endif
    return found;
}

int64_t MmapSnapshotWriter::count_frames(const std::string& filename) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return -1;
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mh == NULL) { CloseHandle(fh); return -1; }
    void* base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) { CloseHandle(mh); CloseHandle(fh); return -1; }
#else
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (::fstat(fd, &st) != 0) { ::close(fd); return -1; }
    void* base = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                        PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { ::close(fd); return -1; }
#endif

    const MmapFileHeader* hdr = reinterpret_cast<const MmapFileHeader*>(base);
    int64_t result = (hdr->magic == MMAP_FILE_MAGIC) ? hdr->frame_count : -1;

#ifdef _WIN32
    UnmapViewOfFile(base); CloseHandle(mh); CloseHandle(fh);
#else
    ::munmap(base, static_cast<size_t>(st.st_size)); ::close(fd);
#endif
    return result;
}

}
