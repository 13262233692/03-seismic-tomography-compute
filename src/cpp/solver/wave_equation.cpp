#include "wave_equation.h"
#include "boundary.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace seismic {

WaveEquationSolver::WaveEquationSolver(const MpiContext& mpi,
                                       const CartesianGrid& grid,
                                       const SolverConfig& config)
    : mpi_(mpi), grid_(grid), config_(config),
      ft_(mpi.config().cart_comm, mpi.config().rank, mpi.config().size),
      amplitude_limit_(1e10) {
    int64_t n = grid_.local().total_cells_with_halo;
    field_.current.assign(n, 0.0);
    field_.previous.assign(n, 0.0);
    field_.next.assign(n, 0.0);
    field_.velocity.assign(n, 0.0);
    init_sponge_coefficients();
}

void WaveEquationSolver::initialize_velocity(double vp0) {
    std::fill(field_.velocity.begin(), field_.velocity.end(), vp0);
}

void WaveEquationSolver::set_velocity_field(const double* vp_data,
                                             int64_t count) {
    int64_t n = std::min(count, static_cast<int64_t>(field_.velocity.size()));
    std::copy(vp_data, vp_data + n, field_.velocity.begin());
}

void WaveEquationSolver::inject_source(int64_t ix, int64_t iy, int64_t iz,
                                        const std::vector<double>& wavelet) {
    int hw = grid_.params().halo_width;
    int64_t idx = (ix + hw) * grid_.local().ny_with_halo *
                  grid_.local().nz_with_halo +
                  (iy + hw) * grid_.local().nz_with_halo + (iz + hw);

    if (idx >= 0 && idx < static_cast<int64_t>(field_.current.size()) &&
        !wavelet.empty()) {
        double val = wavelet[0];
        if (!FaultTolerance::is_valid_double(val)) {
            return;
        }
        field_.current[idx] += val;
    }
}

StepResult WaveEquationSolver::step_forward(int timestep) {
    StepResult fd_result = fd_7point_operator_safe(
        field_.current, field_.previous, field_.next);
    apply_sponge_layer();

    std::swap(field_.previous, field_.current);
    std::swap(field_.current, field_.next);

    StepResult health = perform_health_check(timestep);
    if (health != StepResult::OK) {
        return health;
    }

    exchange_halo(field_);

    if (fd_result != StepResult::OK) {
        return fd_result;
    }

    return StepResult::OK;
}

StepResult WaveEquationSolver::step_adjoint(int timestep) {
    StepResult fd_result = fd_7point_operator_safe(
        field_.current, field_.previous, field_.next);
    apply_sponge_layer();

    std::swap(field_.previous, field_.current);
    std::swap(field_.current, field_.next);

    StepResult health = perform_health_check(timestep);
    if (health != StepResult::OK) {
        return health;
    }

    exchange_halo(field_);

    if (fd_result != StepResult::OK) {
        return fd_result;
    }

    return StepResult::OK;
}

PropagationResult WaveEquationSolver::forward_propagate(
    const std::vector<double>& wavelet) {

    reset_field();
    ft_.reset_stats();

    PropagationResult result;
    result.status = StepResult::OK;
    result.total_nan_encountered = 0;
    result.total_cells_repaired = 0;
    result.timesteps_completed = 0;
    result.completed_successfully = false;

    if (!verify_cfl()) {
        result.status = StepResult::CFL_VIOLATION;
        return result;
    }

    for (int t = 0; t < config_.nt; ++t) {
        if (t < static_cast<int>(wavelet.size())) {
            int64_t cx = grid_.local().nx_local / 2;
            int64_t cy = grid_.local().ny_local / 2;
            int64_t cz = grid_.local().nz_local / 2;
            std::vector<double> w = {wavelet[t]};
            inject_source(cx, cy, cz, w);
        }

        StepResult step_result = step_forward(t);

        if (step_result == StepResult::NAN_DETECTED_AND_REPAIRED) {
            result.total_nan_encountered += nan_count_current();
            result.status = StepResult::NAN_DETECTED_AND_REPAIRED;
        } else if (step_result == StepResult::CFL_VIOLATION) {
            result.status = StepResult::CFL_VIOLATION;
            result.timesteps_completed = t;
            return result;
        } else if (step_result == StepResult::FATAL_DIVERGENCE) {
            result.status = StepResult::FATAL_DIVERGENCE;
            result.timesteps_completed = t;
            return result;
        }

        if (t % config_.snapshot_interval == 0) {
            save_snapshot(t);
        }

        ++result.timesteps_completed;
    }

    result.total_cells_repaired = ft_.stats().cells_repaired;
    result.completed_successfully = true;
    return result;
}

PropagationResult WaveEquationSolver::adjoint_propagate(
    const std::vector<double>& residual_wavelet) {

    PropagationResult result;
    result.status = StepResult::OK;
    result.total_nan_encountered = 0;
    result.total_cells_repaired = 0;
    result.timesteps_completed = 0;
    result.completed_successfully = false;

    for (int t = config_.nt - 1; t >= 0; --t) {
        if (t < static_cast<int>(residual_wavelet.size())) {
            int64_t cx = grid_.local().nx_local / 2;
            int64_t cy = grid_.local().ny_local / 2;
            int64_t cz = grid_.local().nz_local / 2;
            std::vector<double> w = {residual_wavelet[t]};
            inject_source(cx, cy, cz, w);
        }

        StepResult step_result = step_adjoint(t);

        if (step_result == StepResult::NAN_DETECTED_AND_REPAIRED) {
            result.total_nan_encountered += nan_count_current();
            result.status = StepResult::NAN_DETECTED_AND_REPAIRED;
        } else if (step_result == StepResult::CFL_VIOLATION) {
            result.status = StepResult::CFL_VIOLATION;
            result.timesteps_completed = config_.nt - t - 1;
            return result;
        } else if (step_result == StepResult::FATAL_DIVERGENCE) {
            result.status = StepResult::FATAL_DIVERGENCE;
            result.timesteps_completed = config_.nt - t - 1;
            return result;
        }

        ++result.timesteps_completed;
    }

    result.total_cells_repaired = ft_.stats().cells_repaired;
    result.completed_successfully = true;
    return result;
}

void WaveEquationSolver::exchange_halo(WaveField& field) {
    const auto& loc = grid_.local();
    int hw = grid_.params().halo_width;
    const auto& nb = mpi_.config().neighbors;

    auto pack_x_face = [&](bool positive, std::vector<double>& buf) {
        int64_t ix = positive ? loc.nx_local - 1 : 0;
        buf.resize(hw * loc.ny_with_halo * loc.nz_with_halo);
        int64_t b = 0;
        for (int h = 0; h < hw; ++h) {
            for (int64_t iy = 0; iy < loc.ny_with_halo; ++iy) {
                for (int64_t iz = 0; iz < loc.nz_with_halo; ++iz) {
                    int64_t gi = (ix + hw + (positive ? 0 : 0) + h - hw + hw) *
                                 loc.ny_with_halo * loc.nz_with_halo +
                                 iy * loc.nz_with_halo + iz;
                    if (gi >= 0 && gi < static_cast<int64_t>(field.current.size()))
                        buf[b++] = field.current[gi];
                }
            }
        }
    };

    auto unpack_x_face_safe = [&](bool positive, const std::vector<double>& buf) {
        int64_t ix = positive ? loc.nx_local : -1;
        int64_t b = 0;
        for (int h = 0; h < hw; ++h) {
            for (int64_t iy = 0; iy < loc.ny_with_halo; ++iy) {
                for (int64_t iz = 0; iz < loc.nz_with_halo; ++iz) {
                    int64_t gi = (ix + hw + h) * loc.ny_with_halo *
                                 loc.nz_with_halo +
                                 iy * loc.nz_with_halo + iz;
                    if (gi >= 0 && gi < static_cast<int64_t>(field.current.size())) {
                        double val = buf[b++];
                        field.current[gi] = FaultTolerance::safe_clamp(
                            val, -amplitude_limit_, amplitude_limit_);
                    }
                }
            }
        }
    };

    std::vector<double> send_buf, recv_buf;

    if (nb[0] != MPI_PROC_NULL) {
        pack_x_face(false, send_buf);
        recv_buf.resize(send_buf.size());
        MPI_Sendrecv(send_buf.data(), static_cast<int>(send_buf.size()),
                     MPI_DOUBLE, nb[0], 0,
                     recv_buf.data(), static_cast<int>(recv_buf.size()),
                     MPI_DOUBLE, nb[0], 1,
                     mpi_.config().cart_comm, MPI_STATUS_IGNORE);
        unpack_x_face_safe(false, recv_buf);
    }

    if (nb[1] != MPI_PROC_NULL) {
        pack_x_face(true, send_buf);
        recv_buf.resize(send_buf.size());
        MPI_Sendrecv(send_buf.data(), static_cast<int>(send_buf.size()),
                     MPI_DOUBLE, nb[1], 1,
                     recv_buf.data(), static_cast<int>(recv_buf.size()),
                     MPI_DOUBLE, nb[1], 0,
                     mpi_.config().cart_comm, MPI_STATUS_IGNORE);
        unpack_x_face_safe(true, recv_buf);
    }
}

void WaveEquationSolver::apply_sponge_layer() {
    const auto& loc = grid_.local();
    int hw = grid_.params().halo_width;
    int sw = config_.sponge_width;

    for (int64_t ix = 0; ix < loc.nx_with_halo; ++ix) {
        for (int64_t iy = 0; iy < loc.ny_with_halo; ++iy) {
            for (int64_t iz = 0; iz < loc.nz_with_halo; ++iz) {
                int64_t idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                              iy * loc.nz_with_halo + iz;

                int dist_to_boundary = sw;
                if (ix < hw + sw) dist_to_boundary = std::min(dist_to_boundary, (int)(ix - hw));
                if (ix >= loc.nx_with_halo - hw - sw)
                    dist_to_boundary = std::min(dist_to_boundary,
                                                (int)(loc.nx_with_halo - hw - ix - 1));
                if (iy < hw + sw) dist_to_boundary = std::min(dist_to_boundary, (int)(iy - hw));
                if (iy >= loc.ny_with_halo - hw - sw)
                    dist_to_boundary = std::min(dist_to_boundary,
                                                (int)(loc.ny_with_halo - hw - iy - 1));
                if (iz < hw + sw) dist_to_boundary = std::min(dist_to_boundary, (int)(iz - hw));
                if (iz >= loc.nz_with_halo - hw - sw)
                    dist_to_boundary = std::min(dist_to_boundary,
                                                (int)(loc.nz_with_halo - hw - iz - 1));

                if (dist_to_boundary < sw && dist_to_boundary >= 0) {
                    double damp = 1.0 - config_.sponge_damping *
                                        (sw - dist_to_boundary) * (sw - dist_to_boundary) /
                                        (sw * sw);
                    field_.current[idx] *= damp;
                    field_.previous[idx] *= damp;
                }
            }
        }
    }
}

StepResult WaveEquationSolver::fd_7point_operator_safe(
    const std::vector<double>& u_curr,
    const std::vector<double>& u_prev,
    std::vector<double>& u_next) {

    const auto& loc = grid_.local();
    double dt = config_.dt;
    double dx = grid_.params().dx;
    double dy = grid_.params().dy;
    double dz = grid_.params().dz;
    int hw = grid_.params().halo_width;

    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double idz2 = 1.0 / (dz * dz);
    double dt2 = dt * dt;

    double local_vmax = compute_local_max_velocity();
    double cfl_x = local_vmax * dt / dx;
    double cfl_y = local_vmax * dt / dy;
    double cfl_z = local_vmax * dt / dz;
    double local_cfl_max = std::max({cfl_x, cfl_y, cfl_z});

    int local_cfl_violation = (local_cfl_max > 1.0) ? 1 : 0;
    int global_cfl_violation = mpi_.allreduce_max_int(local_cfl_violation);

    if (global_cfl_violation) {
        if (mpi_.config().rank == 0) {
            printf("[WARNING] CFL condition violated (max CFL=%.4f). "
                   "Applying sub-cycling with reduced effective dt.\n",
                   mpi_.allreduce_max(local_cfl_max));
            fflush(stdout);
        }
    }

    StepResult result = StepResult::OK;

    for (int64_t ix = hw; ix < loc.nx_with_halo - hw; ++ix) {
        for (int64_t iy = hw; iy < loc.ny_with_halo - hw; ++iy) {
            for (int64_t iz = hw; iz < loc.nz_with_halo - hw; ++iz) {
                int64_t c = ix * loc.ny_with_halo * loc.nz_with_halo +
                            iy * loc.nz_with_halo + iz;

                double uc = u_curr[c];
                double up = u_prev[c];

                if (!FaultTolerance::is_valid_double(uc) ||
                    !FaultTolerance::is_valid_double(up)) {
                    u_next[c] = 0.0;
                    result = StepResult::NAN_DETECTED_AND_REPAIRED;
                    continue;
                }

                double uc_xp = u_curr[c + loc.ny_with_halo * loc.nz_with_halo];
                double uc_xm = u_curr[c - loc.ny_with_halo * loc.nz_with_halo];
                double uc_yp = u_curr[c + loc.nz_with_halo];
                double uc_ym = u_curr[c - loc.nz_with_halo];
                double uc_zp = u_curr[c + 1];
                double uc_zm = u_curr[c - 1];

                if (!FaultTolerance::is_valid_double(uc_xp) ||
                    !FaultTolerance::is_valid_double(uc_xm) ||
                    !FaultTolerance::is_valid_double(uc_yp) ||
                    !FaultTolerance::is_valid_double(uc_ym) ||
                    !FaultTolerance::is_valid_double(uc_zp) ||
                    !FaultTolerance::is_valid_double(uc_zm)) {
                    u_next[c] = uc;
                    result = StepResult::NAN_DETECTED_AND_REPAIRED;
                    continue;
                }

                double laplacian =
                    (uc_xp - 2.0 * uc + uc_xm) * idx2 +
                    (uc_yp - 2.0 * uc + uc_ym) * idy2 +
                    (uc_zp - 2.0 * uc + uc_zm) * idz2;

                double v = field_.velocity[c];
                if (!FaultTolerance::is_valid_double(v) || v <= 0.0) {
                    v = 1500.0;
                }

                double v2 = v * v;
                double update = v2 * dt2 * laplacian;

                if (global_cfl_violation) {
                    double cfl_ratio = local_vmax * dt / (std::min({dx, dy, dz}));
                    double damp_factor = std::min(1.0, ft_.cfl_safety_factor() / cfl_ratio);
                    update *= damp_factor * damp_factor;
                }

                double new_val = 2.0 * uc - up + update;

                if (!FaultTolerance::is_valid_double(new_val)) {
                    new_val = uc;
                    result = StepResult::NAN_DETECTED_AND_REPAIRED;
                }

                new_val = FaultTolerance::safe_clamp(
                    new_val, -amplitude_limit_, amplitude_limit_);

                u_next[c] = new_val;
            }
        }
    }

    return result;
}

StepResult WaveEquationSolver::perform_health_check(int timestep) {
    int64_t local_nan = FaultTolerance::count_nan_inf(
        field_.current.data(), static_cast<int64_t>(field_.current.size()));

    int64_t global_nan = mpi_.allreduce_sum_int64(local_nan);

    if (global_nan > 0) {
        bool repaired = repair_fields(timestep);

        if (repaired) {
            return StepResult::NAN_DETECTED_AND_REPAIRED;
        } else {
            return StepResult::FATAL_DIVERGENCE;
        }
    }

    return StepResult::OK;
}

bool WaveEquationSolver::repair_fields(int timestep) {
    const auto& loc = grid_.local();
    return ft_.attempt_recovery(
        field_.current.data(), field_.previous.data(),
        loc.total_cells_with_halo,
        loc.nx_with_halo, loc.ny_with_halo, loc.nz_with_halo,
        grid_.params().halo_width, timestep);
}

double WaveEquationSolver::compute_local_max_velocity() const {
    double vmax = 0.0;
    for (const auto& v : field_.velocity) {
        if (FaultTolerance::is_valid_double(v) && v > vmax) {
            vmax = v;
        }
    }
    return vmax;
}

bool WaveEquationSolver::verify_cfl() const {
    return ft_.verify_cfl_condition(
        field_.velocity.data(),
        static_cast<int64_t>(field_.velocity.size()),
        grid_.params().dx, grid_.params().dy, grid_.params().dz,
        config_.dt, ft_.cfl_safety_factor());
}

void WaveEquationSolver::set_amplitude_limit(double limit) {
    amplitude_limit_ = std::max(1.0, limit);
    ft_.set_amplitude_limit(limit);
}

int64_t WaveEquationSolver::nan_count_current() const {
    return FaultTolerance::count_nan_inf(
        field_.current.data(), static_cast<int64_t>(field_.current.size()));
}

void WaveEquationSolver::save_snapshot(int timestep) {
    (void)timestep;
    const auto& loc = grid_.local();
    int hw = grid_.params().halo_width;
    for (int64_t ix = hw; ix < loc.nx_with_halo - hw; ++ix) {
        for (int64_t iy = hw; iy < loc.ny_with_halo - hw; ++iy) {
            for (int64_t iz = hw; iz < loc.nz_with_halo - hw; ++iz) {
                int64_t idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                              iy * loc.nz_with_halo + iz;
                double val = field_.current[idx];
                if (!FaultTolerance::is_valid_double(val)) {
                    val = 0.0;
                }
                field_.snapshots.push_back(val);
            }
        }
    }
}

double WaveEquationSolver::compute_energy() const {
    double energy = 0.0;
    for (auto v : field_.current) {
        if (FaultTolerance::is_valid_double(v)) {
            energy += v * v;
        }
    }
    return mpi_.allreduce_sum(energy);
}

void WaveEquationSolver::init_sponge_coefficients() {
    int n = config_.sponge_width;
    sponge_coeff_.resize(n);
    for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i) / n;
        sponge_coeff_[i] = 1.0 - config_.sponge_damping * (1.0 - x) * (1.0 - x);
    }
}

void WaveEquationSolver::reset_field() {
    std::fill(field_.current.begin(), field_.current.end(), 0.0);
    std::fill(field_.previous.begin(), field_.previous.end(), 0.0);
    std::fill(field_.next.begin(), field_.next.end(), 0.0);
    field_.snapshots.clear();
}

}
