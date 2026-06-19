#pragma once

#include <mpi.h>
#include <vector>
#include <cstdint>
#include <string>
#include <functional>
#include <cmath>
#include <limits>

namespace seismic {

enum class HealthStatus : int {
    HEALTHY = 0,
    NAN_DETECTED = 1,
    INF_DETECTED = 2,
    CFL_VIOLATION = 3,
    DIVERGENCE = 4,
    COMM_FAILURE = 5
};

struct FaultRecord {
    int rank;
    HealthStatus status;
    int64_t timestep;
    int64_t cell_index;
    double offending_value;
    double velocity_at_fault;
};

class FaultTolerance {
public:
    FaultTolerance(MPI_Comm comm, int rank, int size);

    int global_health_check(HealthStatus local_status) const;
    int global_health_check_field(const double* data, int64_t count) const;

    int broadcast_error(int local_error) const;
    void synchronized_abort(int error_code, const std::string& message) const;

    bool verify_cfl_condition(const double* velocity, int64_t count,
                               double dx, double dy, double dz,
                               double dt, double cfl_limit = 0.5) const;

    static int64_t count_nan_inf(const double* data, int64_t count);
    static bool is_valid_double(double val);

    static double safe_clamp(double val, double lo, double hi);
    static void clamp_field(double* data, int64_t count, double lo, double hi);

    void repair_field_nan(double* data, int64_t count,
                          int64_t nx, int64_t ny, int64_t nz,
                          int halo_width, double replace_value = 0.0) const;

    void identify_corrupted_ranks(const std::vector<int>& rank_statuses,
                                   std::vector<int>& corrupted_ranks) const;

    void build_exclusion_mask(const std::vector<int>& corrupted_ranks,
                               std::vector<uint8_t>& mask) const;

    struct FaultStats {
        int64_t total_nan_count;
        int64_t total_inf_count;
        int64_t cells_repaired;
        int64_t timesteps_recovered;
        int global_abort_count;
    };

    const FaultStats& stats() const { return stats_; }
    void reset_stats();

    void set_max_repair_attempts(int max_attempts);
    int max_repair_attempts() const { return max_repair_attempts_; }

    void set_amplitude_limit(double limit);
    double amplitude_limit() const { return amplitude_limit_; }

    void set_cfl_safety_factor(double factor);
    double cfl_safety_factor() const { return cfl_safety_factor_; }

    bool attempt_recovery(double* field_current, double* field_previous,
                           int64_t count, int64_t nx, int64_t ny, int64_t nz,
                           int halo_width, int64_t timestep);

    void log_fault(const FaultRecord& record);
    const std::vector<FaultRecord>& fault_log() const { return fault_log_; }

private:
    MPI_Comm comm_;
    int rank_;
    int size_;
    FaultStats stats_;
    int max_repair_attempts_;
    double amplitude_limit_;
    double cfl_safety_factor_;
    std::vector<FaultRecord> fault_log_;
};

}
