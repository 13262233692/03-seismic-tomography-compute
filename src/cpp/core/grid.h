#pragma once

#include <cstdint>
#include <vector>
#include <array>

namespace seismic {

struct GridParams {
    int64_t nx_global;
    int64_t ny_global;
    int64_t nz_global;
    double dx;
    double dy;
    double dz;
    double origin_x;
    double origin_y;
    double origin_z;
    int halo_width;
};

struct LocalGrid {
    int64_t nx_local;
    int64_t ny_local;
    int64_t nz_local;
    int64_t nx_with_halo;
    int64_t ny_with_halo;
    int64_t nz_with_halo;
    int64_t offset_x;
    int64_t offset_y;
    int64_t offset_z;
    int64_t total_cells;
    int64_t total_cells_with_halo;
};

class CartesianGrid {
public:
    CartesianGrid(const GridParams& params, const int mpi_coords[3],
                  const int mpi_dims[3]);

    int64_t local_index(int64_t ix, int64_t iy, int64_t iz) const;
    int64_t local_index_with_halo(int64_t ix, int64_t iy, int64_t iz) const;

    void global_to_local(int64_t gx, int64_t gy, int64_t gz,
                         int64_t& lx, int64_t& ly, int64_t& lz) const;

    void local_to_global(int64_t lx, int64_t ly, int64_t lz,
                         int64_t& gx, int64_t& gy, int64_t& gz) const;

    bool is_interior(int64_t lx, int64_t ly, int64_t lz) const;
    bool is_halo(int64_t lx, int64_t ly, int64_t lz) const;

    const GridParams& params() const { return params_; }
    const LocalGrid& local() const { return local_; }

    std::vector<double> compute_coordinates() const;

    double x_coord(int64_t ix) const;
    double y_coord(int64_t iy) const;
    double z_coord(int64_t iz) const;

private:
    GridParams params_;
    LocalGrid local_;

    void compute_local_decomposition(const int mpi_coords[3],
                                     const int mpi_dims[3]);
};

}
