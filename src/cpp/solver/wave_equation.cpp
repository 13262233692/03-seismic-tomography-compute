#include "wave_equation.h"
#include "boundary.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace seismic {

WaveEquationSolver::WaveEquationSolver(const MpiContext& mpi,
                                       const CartesianGrid& grid,
                                       const SolverConfig& config)
    : mpi_(mpi), grid_(grid), config_(config) {
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
        field_.current[idx] += wavelet[0];
    }
}

void WaveEquationSolver::step_forward(int timestep) {
    (void)timestep;
    fd_7point_operator(field_.current, field_.previous, field_.next);
    apply_sponge_layer();

    std::swap(field_.previous, field_.current);
    std::swap(field_.current, field_.next);

    exchange_halo(field_);
}

void WaveEquationSolver::step_adjoint(int timestep) {
    (void)timestep;
    fd_7point_operator(field_.current, field_.previous, field_.next);
    apply_sponge_layer();

    std::swap(field_.previous, field_.current);
    std::swap(field_.current, field_.next);

    exchange_halo(field_);
}

void WaveEquationSolver::forward_propagate(
    const std::vector<double>& wavelet) {
    reset_field();
    for (int t = 0; t < config_.nt; ++t) {
        if (t < static_cast<int>(wavelet.size())) {
            int64_t cx = grid_.local().nx_local / 2;
            int64_t cy = grid_.local().ny_local / 2;
            int64_t cz = grid_.local().nz_local / 2;
            std::vector<double> w = {wavelet[t]};
            inject_source(cx, cy, cz, w);
        }
        step_forward(t);
        if (t % config_.snapshot_interval == 0) {
            save_snapshot(t);
        }
    }
}

void WaveEquationSolver::adjoint_propagate(
    const std::vector<double>& residual_wavelet) {
    for (int t = config_.nt - 1; t >= 0; --t) {
        if (t < static_cast<int>(residual_wavelet.size())) {
            int64_t cx = grid_.local().nx_local / 2;
            int64_t cy = grid_.local().ny_local / 2;
            int64_t cz = grid_.local().nz_local / 2;
            std::vector<double> w = {residual_wavelet[t]};
            inject_source(cx, cy, cz, w);
        }
        step_adjoint(t);
    }
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

    auto unpack_x_face = [&](bool positive, const std::vector<double>& buf) {
        int64_t ix = positive ? loc.nx_local : -1;
        int64_t b = 0;
        for (int h = 0; h < hw; ++h) {
            for (int64_t iy = 0; iy < loc.ny_with_halo; ++iy) {
                for (int64_t iz = 0; iz < loc.nz_with_halo; ++iz) {
                    int64_t gi = (ix + hw + h) * loc.ny_with_halo *
                                 loc.nz_with_halo +
                                 iy * loc.nz_with_halo + iz;
                    if (gi >= 0 && gi < static_cast<int64_t>(field.current.size()))
                        field.current[gi] = buf[b++];
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
        unpack_x_face(false, recv_buf);
    }

    if (nb[1] != MPI_PROC_NULL) {
        pack_x_face(true, send_buf);
        recv_buf.resize(send_buf.size());
        MPI_Sendrecv(send_buf.data(), static_cast<int>(send_buf.size()),
                     MPI_DOUBLE, nb[1], 1,
                     recv_buf.data(), static_cast<int>(recv_buf.size()),
                     MPI_DOUBLE, nb[1], 0,
                     mpi_.config().cart_comm, MPI_STATUS_IGNORE);
        unpack_x_face(true, recv_buf);
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

void WaveEquationSolver::fd_7point_operator(
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

    for (int64_t ix = hw; ix < loc.nx_with_halo - hw; ++ix) {
        for (int64_t iy = hw; iy < loc.ny_with_halo - hw; ++iy) {
            for (int64_t iz = hw; iz < loc.nz_with_halo - hw; ++iz) {
                int64_t c = ix * loc.ny_with_halo * loc.nz_with_halo +
                            iy * loc.nz_with_halo + iz;

                double laplacian =
                    (u_curr[c + loc.ny_with_halo * loc.nz_with_halo] -
                     2.0 * u_curr[c] +
                     u_curr[c - loc.ny_with_halo * loc.nz_with_halo]) * idx2 +
                    (u_curr[c + loc.nz_with_halo] - 2.0 * u_curr[c] +
                     u_curr[c - loc.nz_with_halo]) * idy2 +
                    (u_curr[c + 1] - 2.0 * u_curr[c] + u_curr[c - 1]) * idz2;

                double v2 = field_.velocity[c] * field_.velocity[c];
                u_next[c] = 2.0 * u_curr[c] - u_prev[c] + v2 * dt2 * laplacian;
            }
        }
    }
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
                field_.snapshots.push_back(field_.current[idx]);
            }
        }
    }
}

double WaveEquationSolver::compute_energy() const {
    double energy = 0.0;
    for (auto v : field_.current) {
        energy += v * v;
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
