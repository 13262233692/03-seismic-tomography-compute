#include "rtm_engine.h"
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace seismic {

RTMEngine::RTMEngine(int64_t nx, int64_t ny, int64_t nz,
                     double dx, double dy, double dz,
                     int halo_width)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz),
      halo_width_(halo_width), slices_captured_(0) {
    int64_t n = nx_ * ny_ * nz_;
    image_.image.assign(n, 0.0);
    image_.illumination.assign(n, 0.0);
    image_.reflectivity.assign(n, 0.0);
}

void RTMEngine::set_slice_specs(const std::vector<SliceSpec>& specs) {
    slice_specs_ = specs;
}

void RTMEngine::apply_image_condition(const double* forward_wavefield,
                                       const double* adjoint_wavefield,
                                       int64_t ny, int64_t nz,
                                       int64_t step_idx) {
    (void)step_idx;

    int64_t hw = halo_width_;
    int64_t nynz = ny * nz;

    int64_t nx_inner = nx_;
    int64_t ny_inner = ny_;
    int64_t nz_inner = nz_;

    for (int64_t ix = 0; ix < nx_inner; ++ix) {
        int64_t ix_hw = ix + hw;
        for (int64_t iy = 0; iy < ny_inner; ++iy) {
            int64_t iy_hw = iy + hw;
            for (int64_t iz = 0; iz < nz_inner; ++iz) {
                int64_t iz_hw = iz + hw;

                int64_t with_halo_idx = ix_hw * nynz + iy_hw * nz + iz_hw;
                int64_t inner_idx = ix * ny_inner * nz_inner + iy * nz_inner + iz;

                double f = std::isfinite(forward_wavefield[with_halo_idx])
                               ? forward_wavefield[with_halo_idx] : 0.0;
                double a = std::isfinite(adjoint_wavefield[with_halo_idx])
                               ? adjoint_wavefield[with_halo_idx] : 0.0;

                image_.image[inner_idx] += f * a;
                image_.illumination[inner_idx] += f * f;
            }
        }
    }
}

void RTMEngine::extract_slices(const double* forward_wavefield,
                                const double* adjoint_wavefield,
                                const double* image_condition,
                                int64_t ny, int64_t nz,
                                int64_t timestep) {
    if (!slice_cb_ || slice_specs_.empty()) return;

    int64_t hw = halo_width_;
    const double* ic = (image_condition != nullptr) ? image_condition : image_.image.data();

    for (const auto& spec : slice_specs_) {
        if (spec.index < 0) {
            switch (spec.axis) {
                case SliceAxis::X: capture_single_slice(spec.axis, nx_ / 2, forward_wavefield, adjoint_wavefield, ic, ny, nz, timestep); break;
                case SliceAxis::Y: capture_single_slice(spec.axis, ny_ / 2, forward_wavefield, adjoint_wavefield, ic, ny, nz, timestep); break;
                case SliceAxis::Z: capture_single_slice(spec.axis, nz_ / 2, forward_wavefield, adjoint_wavefield, ic, ny, nz, timestep); break;
                default: break;
            }
        } else {
            capture_single_slice(spec.axis, spec.index, forward_wavefield, adjoint_wavefield, ic, ny, nz, timestep);
        }
    }
}

void RTMEngine::capture_single_slice(SliceAxis axis, int64_t slice_idx,
                                      const double* fwd, const double* adj, const double* ic,
                                      int64_t ny, int64_t nz, int64_t timestep) {
    int64_t hw = halo_width_;
    int64_t nynz = ny * nz;

    CapturedSlice slice;
    slice.timestep = timestep;
    slice.axis = axis;
    slice.index = slice_idx;

    switch (axis) {
        case SliceAxis::X: {
            slice.dim0 = ny_;
            slice.dim1 = nz_;
            slice.forward_slice.resize(slice.dim0 * slice.dim1);
            slice.adjoint_slice.resize(slice.dim0 * slice.dim1);
            slice.ic_slice.resize(slice.dim0 * slice.dim1);

            int64_t ix_hw = slice_idx + hw;
            for (int64_t iy = 0; iy < ny_; ++iy) {
                int64_t iy_hw = iy + hw;
                for (int64_t iz = 0; iz < nz_; ++iz) {
                    int64_t iz_hw = iz + hw;
                    int64_t src_idx = ix_hw * nynz + iy_hw * nz + iz_hw;
                    int64_t dst_idx = iy * nz_ + iz;
                    slice.forward_slice[dst_idx] = std::isfinite(fwd[src_idx]) ? fwd[src_idx] : 0.0;
                    slice.adjoint_slice[dst_idx] = std::isfinite(adj[src_idx]) ? adj[src_idx] : 0.0;
                    slice.ic_slice[dst_idx] = std::isfinite(ic[iy * nz_ + iz]) ? ic[iy * nz_ + iz] : 0.0;
                }
            }
            break;
        }
        case SliceAxis::Y: {
            slice.dim0 = nx_;
            slice.dim1 = nz_;
            slice.forward_slice.resize(slice.dim0 * slice.dim1);
            slice.adjoint_slice.resize(slice.dim0 * slice.dim1);
            slice.ic_slice.resize(slice.dim0 * slice.dim1);

            int64_t iy_hw = slice_idx + hw;
            for (int64_t ix = 0; ix < nx_; ++ix) {
                int64_t ix_hw = ix + hw;
                for (int64_t iz = 0; iz < nz_; ++iz) {
                    int64_t iz_hw = iz + hw;
                    int64_t src_idx = ix_hw * nynz + iy_hw * nz + iz_hw;
                    int64_t dst_idx = ix * nz_ + iz;
                    slice.forward_slice[dst_idx] = std::isfinite(fwd[src_idx]) ? fwd[src_idx] : 0.0;
                    slice.adjoint_slice[dst_idx] = std::isfinite(adj[src_idx]) ? adj[src_idx] : 0.0;
                    slice.ic_slice[dst_idx] = std::isfinite(ic[ix * nz_ + iz]) ? ic[ix * nz_ + iz] : 0.0;
                }
            }
            break;
        }
        case SliceAxis::Z: {
            slice.dim0 = nx_;
            slice.dim1 = ny_;
            slice.forward_slice.resize(slice.dim0 * slice.dim1);
            slice.adjoint_slice.resize(slice.dim0 * slice.dim1);
            slice.ic_slice.resize(slice.dim0 * slice.dim1);

            int64_t iz_hw = slice_idx + hw;
            for (int64_t ix = 0; ix < nx_; ++ix) {
                int64_t ix_hw = ix + hw;
                for (int64_t iy = 0; iy < ny_; ++iy) {
                    int64_t iy_hw = iy + hw;
                    int64_t src_idx = ix_hw * nynz + iy_hw * nz + iz_hw;
                    int64_t dst_idx = ix * ny_ + iy;
                    slice.forward_slice[dst_idx] = std::isfinite(fwd[src_idx]) ? fwd[src_idx] : 0.0;
                    slice.adjoint_slice[dst_idx] = std::isfinite(adj[src_idx]) ? adj[src_idx] : 0.0;
                    slice.ic_slice[dst_idx] = std::isfinite(ic[ix * ny_ + iy]) ? ic[ix * ny_ + iy] : 0.0;
                }
            }
            break;
        }
        default:
            return;
    }

    auto now = std::chrono::high_resolution_clock::now();
    slice.capture_timestamp = std::chrono::duration<double>(
        now.time_since_epoch()).count();

    slice_cb_(slice);
    slices_captured_.fetch_add(1, std::memory_order_release);
}

void RTMEngine::finalize_image() {
    int64_t n = total_cells();
    const double eps = 1e-10;

    for (int64_t i = 0; i < n; ++i) {
        double illum = image_.illumination[i];
        if (illum > eps) {
            image_.reflectivity[i] = image_.image[i] / illum;
        } else {
            image_.reflectivity[i] = 0.0;
        }
    }
}

void RTMEngine::reset_image() {
    int64_t n = total_cells();
    std::fill(image_.image.begin(), image_.image.end(), 0.0);
    std::fill(image_.illumination.begin(), image_.illumination.end(), 0.0);
    std::fill(image_.reflectivity.begin(), image_.reflectivity.end(), 0.0);
    slices_captured_.store(0, std::memory_order_release);
}

}
