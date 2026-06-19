#include "fault_tolerance.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace seismic {

FaultTolerance::FaultTolerance(MPI_Comm comm, int rank, int size)
    : comm_(comm), rank_(rank), size_(size),
      max_repair_attempts_(3), amplitude_limit_(1e10),
      cfl_safety_factor_(0.49) {
    std::memset(&stats_, 0, sizeof(stats_));
}

int FaultTolerance::global_health_check(HealthStatus local_status) const {
    int local_code = static_cast<int>(local_status);
    int global_code = 0;
    MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX, comm_);
    return global_code;
}

int FaultTolerance::global_health_check_field(const double* data,
                                                int64_t count) const {
    int64_t local_nan_count = count_nan_inf(data, count);
    int local_flag = (local_nan_count > 0) ? 1 : 0;

    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, comm_);
    return global_flag;
}

int FaultTolerance::broadcast_error(int local_error) const {
    int global_error = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, comm_);
    return global_error;
}

void FaultTolerance::synchronized_abort(int error_code,
                                          const std::string& message) const {
    fprintf(stderr, "[Rank %d] FATAL: %s (error_code=%d)\n",
            rank_, message.c_str(), error_code);
    fflush(stderr);

    MPI_Barrier(comm_);
    MPI_Abort(comm_, error_code);
}

bool FaultTolerance::verify_cfl_condition(const double* velocity,
                                            int64_t count,
                                            double dx, double dy, double dz,
                                            double dt,
                                            double cfl_limit) const {
    double v_max = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        if (is_valid_double(velocity[i]) && velocity[i] > v_max) {
            v_max = velocity[i];
        }
    }

    double global_vmax = 0.0;
    MPI_Allreduce(&v_max, &global_vmax, 1, MPI_DOUBLE, MPI_MAX, comm_);

    double cfl_x = global_vmax * dt / dx;
    double cfl_y = global_vmax * dt / dy;
    double cfl_z = global_vmax * dt / dz;
    double cfl_max = std::max({cfl_x, cfl_y, cfl_z});

    return cfl_max <= cfl_limit;
}

int64_t FaultTolerance::count_nan_inf(const double* data, int64_t count) {
    int64_t n = 0;
    for (int64_t i = 0; i < count; ++i) {
        if (!is_valid_double(data[i])) {
            ++n;
        }
    }
    return n;
}

bool FaultTolerance::is_valid_double(double val) {
    return std::isfinite(val);
}

double FaultTolerance::safe_clamp(double val, double lo, double hi) {
    if (!is_valid_double(val)) return 0.0;
    return std::max(lo, std::min(val, hi));
}

void FaultTolerance::clamp_field(double* data, int64_t count,
                                  double lo, double hi) {
    for (int64_t i = 0; i < count; ++i) {
        data[i] = safe_clamp(data[i], lo, hi);
    }
}

void FaultTolerance::repair_field_nan(double* data, int64_t count,
                                       int64_t nx, int64_t ny, int64_t nz,
                                       int halo_width,
                                       double replace_value) const {
    int64_t repaired = 0;
    int64_t nynz = ny * nz;
    int64_t nz_val = nz;

    for (int64_t ix = halo_width; ix < nx - halo_width; ++ix) {
        for (int64_t iy = halo_width; iy < ny - halo_width; ++iy) {
            for (int64_t iz = halo_width; iz < nz - halo_width; ++iz) {
                int64_t idx = ix * nynz + iy * nz_val + iz;

                if (!is_valid_double(data[idx])) {
                    double sum = 0.0;
                    int valid_neighbors = 0;

                    if (ix > halo_width && is_valid_double(data[(ix-1)*nynz + iy*nz_val + iz])) {
                        sum += data[(ix-1)*nynz + iy*nz_val + iz];
                        ++valid_neighbors;
                    }
                    if (ix < nx - halo_width - 1 && is_valid_double(data[(ix+1)*nynz + iy*nz_val + iz])) {
                        sum += data[(ix+1)*nynz + iy*nz_val + iz];
                        ++valid_neighbors;
                    }
                    if (iy > halo_width && is_valid_double(data[ix*nynz + (iy-1)*nz_val + iz])) {
                        sum += data[ix*nynz + (iy-1)*nz_val + iz];
                        ++valid_neighbors;
                    }
                    if (iy < ny - halo_width - 1 && is_valid_double(data[ix*nynz + (iy+1)*nz_val + iz])) {
                        sum += data[ix*nynz + (iy+1)*nz_val + iz];
                        ++valid_neighbors;
                    }
                    if (iz > halo_width && is_valid_double(data[ix*nynz + iy*nz_val + iz - 1])) {
                        sum += data[ix*nynz + iy*nz_val + iz - 1];
                        ++valid_neighbors;
                    }
                    if (iz < nz - halo_width - 1 && is_valid_double(data[ix*nynz + iy*nz_val + iz + 1])) {
                        sum += data[ix*nynz + iy*nz_val + iz + 1];
                        ++valid_neighbors;
                    }

                    if (valid_neighbors > 0) {
                        data[idx] = sum / valid_neighbors;
                    } else {
                        data[idx] = replace_value;
                    }
                    ++repaired;
                }
            }
        }
    }

    for (int64_t i = 0; i < count; ++i) {
        if (!is_valid_double(data[i])) {
            data[i] = replace_value;
            ++repaired;
        }
    }
}

void FaultTolerance::identify_corrupted_ranks(
    const std::vector<int>& rank_statuses,
    std::vector<int>& corrupted_ranks) const {
    corrupted_ranks.clear();
    for (size_t i = 0; i < rank_statuses.size(); ++i) {
        if (rank_statuses[i] != 0) {
            corrupted_ranks.push_back(static_cast<int>(i));
        }
    }
}

void FaultTolerance::build_exclusion_mask(
    const std::vector<int>& corrupted_ranks,
    std::vector<uint8_t>& mask) const {
    mask.assign(size_, 1);
    for (int r : corrupted_ranks) {
        if (r >= 0 && r < size_) {
            mask[r] = 0;
        }
    }
}

void FaultTolerance::reset_stats() {
    std::memset(&stats_, 0, sizeof(stats_));
    fault_log_.clear();
}

void FaultTolerance::set_max_repair_attempts(int max_attempts) {
    max_repair_attempts_ = std::max(1, max_attempts);
}

void FaultTolerance::set_amplitude_limit(double limit) {
    amplitude_limit_ = std::max(1.0, limit);
}

void FaultTolerance::set_cfl_safety_factor(double factor) {
    cfl_safety_factor_ = std::max(0.01, std::min(factor, 0.99));
}

bool FaultTolerance::attempt_recovery(double* field_current,
                                       double* field_previous,
                                       int64_t count,
                                       int64_t nx, int64_t ny, int64_t nz,
                                       int halo_width,
                                       int64_t timestep) {
    int64_t local_nan = count_nan_inf(field_current, count);
    int64_t global_nan = 0;
    MPI_Allreduce(&local_nan, &global_nan, 1, MPI_LONG_LONG, MPI_SUM, comm_);

    if (global_nan == 0) {
        return true;
    }

    stats_.total_nan_count += global_nan;

    FaultRecord rec;
    rec.rank = rank_;
    rec.status = HealthStatus::NAN_DETECTED;
    rec.timestep = timestep;
    rec.cell_index = -1;
    rec.offending_value = 0.0;
    rec.velocity_at_fault = 0.0;

    for (int64_t i = 0; i < count; ++i) {
        if (!is_valid_double(field_current[i])) {
            rec.cell_index = i;
            rec.offending_value = field_current[i];
            if (i < count) {
                rec.velocity_at_fault = 0.0;
            }
            break;
        }
    }
    log_fault(rec);

    for (int attempt = 0; attempt < max_repair_attempts_; ++attempt) {
        repair_field_nan(field_current, count, nx, ny, nz, halo_width, 0.0);
        repair_field_nan(field_previous, count, nx, ny, nz, halo_width, 0.0);

        clamp_field(field_current, count, -amplitude_limit_, amplitude_limit_);
        clamp_field(field_previous, count, -amplitude_limit_, amplitude_limit_);

        int64_t post_repair_nan = count_nan_inf(field_current, count);
        int64_t global_post_nan = 0;
        MPI_Allreduce(&post_repair_nan, &global_post_nan, 1,
                      MPI_LONG_LONG, MPI_SUM, comm_);

        stats_.cells_repaired += (global_nan - global_post_nan);

        if (global_post_nan == 0) {
            stats_.timesteps_recovered++;
            return true;
        }

        global_nan = global_post_nan;
    }

    return false;
}

void FaultTolerance::log_fault(const FaultRecord& record) {
    fault_log_.push_back(record);
}

}
