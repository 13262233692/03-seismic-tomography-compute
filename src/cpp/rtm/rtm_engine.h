#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <condition_variable>

namespace seismic {

enum class SliceAxis {
    X = 0,
    Y = 1,
    Z = 2,
    VOLUME = 3
};

struct SliceSpec {
    SliceAxis axis;
    int64_t index;
    int64_t stride;
};

struct RTMImageCondition {
    std::vector<double> image;
    std::vector<double> illumination;
    std::vector<double> reflectivity;
};

class RTMEngine {
public:
    RTMEngine(int64_t nx, int64_t ny, int64_t nz,
              double dx, double dy, double dz,
              int halo_width);

    void set_slice_specs(const std::vector<SliceSpec>& specs);

    void apply_image_condition(const double* forward_wavefield,
                                const double* adjoint_wavefield,
                                int64_t ny, int64_t nz,
                                int64_t step_idx);

    void extract_slices(const double* forward_wavefield,
                        const double* adjoint_wavefield,
                        const double* image_condition,
                        int64_t ny, int64_t nz,
                        int64_t timestep);

    void finalize_image();

    const RTMImageCondition& result() const { return image_; }

    int64_t nx() const { return nx_; }
    int64_t ny() const { return ny_; }
    int64_t nz() const { return nz_; }

    int64_t total_cells() const { return nx_ * ny_ * nz_; }

    void reset_image();

    struct CapturedSlice {
        int64_t timestep;
        SliceAxis axis;
        int64_t index;
        int64_t dim0;
        int64_t dim1;
        std::vector<double> forward_slice;
        std::vector<double> adjoint_slice;
        std::vector<double> ic_slice;
        double capture_timestamp;
    };

    using SliceCaptureCallback = std::function<void(const CapturedSlice&)>;
    void set_slice_callback(SliceCaptureCallback cb) { slice_cb_ = cb; }

    int64_t total_captured_slices() const { return slices_captured_.load(std::memory_order_acquire); }

private:
    int64_t nx_;
    int64_t ny_;
    int64_t nz_;
    double dx_;
    double dy_;
    double dz_;
    int halo_width_;

    RTMImageCondition image_;
    std::vector<SliceSpec> slice_specs_;
    std::atomic<int64_t> slices_captured_;
    SliceCaptureCallback slice_cb_;

    void capture_single_slice(SliceAxis axis, int64_t slice_idx,
                               const double* fwd, const double* adj, const double* ic,
                               int64_t ny, int64_t nz, int64_t timestep);
};

}
