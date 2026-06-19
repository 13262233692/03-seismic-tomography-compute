#pragma once

#include "core/grid.h"
#include "core/config.h"
#include "core/mpi_context.h"
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

class WaveEquationSolver {
public:
    WaveEquationSolver(const MpiContext& mpi,
                       const CartesianGrid& grid,
                       const SolverConfig& config);

    void initialize_velocity(double vp0);
    void set_velocity_field(const double* vp_data, int64_t count);

    void inject_source(int64_t ix, int64_t iy, int64_t iz,
                       const std::vector<double>& wavelet);

    void step_forward(int timestep);
    void step_adjoint(int timestep);

    void forward_propagate(const std::vector<double>& wavelet);
    void adjoint_propagate(const std::vector<double>& residual_wavelet);

    void exchange_halo(WaveField& field);
    void apply_sponge_layer();

    const WaveField& field() const { return field_; }
    WaveField& mutable_field() { return field_; }

    const CartesianGrid& grid() const { return grid_; }

    void save_snapshot(int timestep);
    void clear_snapshots() { field_.snapshots.clear(); }

    double compute_energy() const;

    void reset_field();

private:
    const MpiContext& mpi_;
    const CartesianGrid& grid_;
    SolverConfig config_;
    WaveField field_;
    std::vector<double> sponge_coeff_;

    void init_sponge_coefficients();
    void fd_7point_operator(const std::vector<double>& u_curr,
                            const std::vector<double>& u_prev,
                            std::vector<double>& u_next);
};

}
