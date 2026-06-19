#pragma once

#include "core/grid.h"
#include "core/config.h"
#include "core/mpi_context.h"
#include "core/fault_tolerance.h"
#include <vector>
#include <memory>

namespace seismic {

struct WaveField {
    std::vector<double> current;
    std::vector<double> previous;
    std::vector<double> next;
    std::vector<double> velocity;
    std::vector<double> snapshots;
};

enum class StepResult {
    OK = 0,
    NAN_DETECTED_AND_REPAIRED = 1,
    CFL_VIOLATION = 2,
    FATAL_DIVERGENCE = 3
};

struct PropagationResult {
    StepResult status;
    int64_t total_nan_encountered;
    int64_t total_cells_repaired;
    int timesteps_completed;
    bool completed_successfully;
};

class WaveEquationSolver {
public:
    WaveEquationSolver(const MpiContext& mpi,
                       const CartesianGrid& grid,
                       const SolverConfig& config);

    void initialize_velocity(double vp0);
    void set_velocity_field(const double* vp_data, int64_t count);

    void inject_source(int64_t ix, int64_t iy, int64_t iz,
                       const std::vector<double>& wavelet);

    StepResult step_forward(int timestep);
    StepResult step_adjoint(int timestep);

    PropagationResult forward_propagate(const std::vector<double>& wavelet);
    PropagationResult adjoint_propagate(const std::vector<double>& residual_wavelet);

    void exchange_halo(WaveField& field);
    void apply_sponge_layer();

    const WaveField& field() const { return field_; }
    WaveField& mutable_field() { return field_; }

    const CartesianGrid& grid() const { return grid_; }

    void save_snapshot(int timestep);
    void clear_snapshots() { field_.snapshots.clear(); }

    double compute_energy() const;

    void reset_field();

    FaultTolerance& fault_tolerance() { return ft_; }
    const FaultTolerance& fault_tolerance() const { return ft_; }

    bool verify_cfl() const;

    void set_amplitude_limit(double limit);
    double amplitude_limit() const { return amplitude_limit_; }

    int64_t nan_count_current() const;

    const FaultTolerance::FaultStats& fault_stats() const { return ft_.stats(); }

private:
    const MpiContext& mpi_;
    const CartesianGrid& grid_;
    SolverConfig config_;
    WaveField field_;
    std::vector<double> sponge_coeff_;
    FaultTolerance ft_;
    double amplitude_limit_;

    void init_sponge_coefficients();

    StepResult fd_7point_operator_safe(const std::vector<double>& u_curr,
                                       const std::vector<double>& u_prev,
                                       std::vector<double>& u_next);

    StepResult perform_health_check(int timestep);

    bool repair_fields(int timestep);

    double compute_local_max_velocity() const;
};

}
