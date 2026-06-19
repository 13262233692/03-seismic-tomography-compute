#include "grid.h"
#include <cmath>
#include <algorithm>

namespace seismic {

CartesianGrid::CartesianGrid(const GridParams& params, const int mpi_coords[3],
                             const int mpi_dims[3])
    : params_(params) {
    compute_local_decomposition(mpi_coords, mpi_dims);
}

void CartesianGrid::compute_local_decomposition(const int mpi_coords[3],
                                                const int mpi_dims[3]) {
    auto decompose_axis = [](int64_t n_global, int n_ranks, int coord,
                            int64_t& n_local, int64_t& offset) {
        int64_t base = n_global / n_ranks;
        int64_t remainder = n_global % n_ranks;
        offset = (int64_t)coord * base + std::min((int64_t)coord, remainder);
        n_local = base + (coord < remainder ? 1 : 0);
    };

    decompose_axis(params_.nx_global, mpi_dims[0], mpi_coords[0],
                   local_.nx_local, local_.offset_x);
    decompose_axis(params_.ny_global, mpi_dims[1], mpi_coords[1],
                   local_.ny_local, local_.offset_y);
    decompose_axis(params_.nz_global, mpi_dims[2], mpi_coords[2],
                   local_.nz_local, local_.offset_z);

    int hw = params_.halo_width;
    local_.nx_with_halo = local_.nx_local + 2 * hw;
    local_.ny_with_halo = local_.ny_local + 2 * hw;
    local_.nz_with_halo = local_.nz_local + 2 * hw;

    local_.total_cells = local_.nx_local * local_.ny_local * local_.nz_local;
    local_.total_cells_with_halo = local_.nx_with_halo *
                                    local_.ny_with_halo *
                                    local_.nz_with_halo;
}

int64_t CartesianGrid::local_index(int64_t ix, int64_t iy, int64_t iz) const {
    return ix * local_.ny_local * local_.nz_local + iy * local_.nz_local + iz;
}

int64_t CartesianGrid::local_index_with_halo(int64_t ix, int64_t iy,
                                              int64_t iz) const {
    return ix * local_.ny_with_halo * local_.nz_with_halo +
           iy * local_.nz_with_halo + iz;
}

void CartesianGrid::global_to_local(int64_t gx, int64_t gy, int64_t gz,
                                    int64_t& lx, int64_t& ly,
                                    int64_t& lz) const {
    lx = gx - local_.offset_x;
    ly = gy - local_.offset_y;
    lz = gz - local_.offset_z;
}

void CartesianGrid::local_to_global(int64_t lx, int64_t ly, int64_t lz,
                                    int64_t& gx, int64_t& gy,
                                    int64_t& gz) const {
    gx = lx + local_.offset_x;
    gy = ly + local_.offset_y;
    gz = lz + local_.offset_z;
}

bool CartesianGrid::is_interior(int64_t lx, int64_t ly, int64_t lz) const {
    return lx >= 0 && lx < local_.nx_local &&
           ly >= 0 && ly < local_.ny_local &&
           lz >= 0 && lz < local_.nz_local;
}

bool CartesianGrid::is_halo(int64_t lx, int64_t ly, int64_t lz) const {
    int hw = params_.halo_width;
    return (lx >= 0 && lx < local_.nx_with_halo &&
            ly >= 0 && ly < local_.ny_with_halo &&
            lz >= 0 && lz < local_.nz_with_halo) &&
           !is_interior(lx - hw, ly - hw, lz - hw);
}

std::vector<double> CartesianGrid::compute_coordinates() const {
    std::vector<double> coords(3 * local_.total_cells);
    int64_t idx = 0;
    for (int64_t ix = 0; ix < local_.nx_local; ++ix) {
        for (int64_t iy = 0; iy < local_.ny_local; ++iy) {
            for (int64_t iz = 0; iz < local_.nz_local; ++iz) {
                coords[3 * idx + 0] = x_coord(ix);
                coords[3 * idx + 1] = y_coord(iy);
                coords[3 * idx + 2] = z_coord(iz);
                ++idx;
            }
        }
    }
    return coords;
}

double CartesianGrid::x_coord(int64_t ix) const {
    return params_.origin_x + (local_.offset_x + ix) * params_.dx;
}

double CartesianGrid::y_coord(int64_t iy) const {
    return params_.origin_y + (local_.offset_y + iy) * params_.dy;
}

double CartesianGrid::z_coord(int64_t iz) const {
    return params_.origin_z + (local_.offset_z + iz) * params_.dz;
}

}
