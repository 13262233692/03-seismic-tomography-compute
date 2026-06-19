#include "boundary.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace seismic {

TerrainBoundary::TerrainBoundary(const CartesianGrid& grid, double water_depth)
    : grid_(grid), water_depth_(water_depth) {
    int64_t nx = grid.params().nx_global;
    int64_t ny = grid.params().ny_global;
    bathymetry_grid_.assign(nx * ny, water_depth);
}

void TerrainBoundary::set_bathymetry_data(
    const std::vector<BathymetryPoint>& data) {
    int64_t nx = grid_.params().nx_global;
    double dx = grid_.params().dx;
    double dy = grid_.params().dy;
    double ox = grid_.params().origin_x;
    double oy = grid_.params().origin_y;

    for (const auto& pt : data) {
        int64_t ix = static_cast<int64_t>((pt.x - ox) / dx);
        int64_t iy = static_cast<int64_t>((pt.y - oy) / dy);
        if (ix >= 0 && ix < nx && iy >= 0 &&
            iy < grid_.params().ny_global) {
            bathymetry_grid_[ix * grid_.params().ny_global + iy] = pt.depth;
        }
    }
}

void TerrainBoundary::generate_synthetic_bathymetry(double gradient,
                                                     double roughness) {
    int64_t nx = grid_.params().nx_global;
    int64_t ny = grid_.params().ny_global;

    for (int64_t ix = 0; ix < nx; ++ix) {
        for (int64_t iy = 0; iy < ny; ++iy) {
            double x = ix * grid_.params().dx;
            double y = iy * grid_.params().dy;

            double depth = water_depth_ +
                           gradient * x +
                           roughness * std::sin(2.0 * M_PI * x / 5000.0) *
                           std::cos(2.0 * M_PI * y / 7000.0) +
                           roughness * 0.5 * std::sin(4.0 * M_PI * x / 3000.0);

            depth = std::max(depth, 500.0);
            depth = std::min(depth, 8000.0);
            bathymetry_grid_[ix * ny + iy] = depth;
        }
    }
}

bool TerrainBoundary::is_subsurface(int64_t gx, int64_t gy,
                                     int64_t gz) const {
    double z_pos = grid_.params().origin_z + gz * grid_.params().dz;
    double sf_depth = seafloor_depth(gx, gy);
    return z_pos >= sf_depth;
}

bool TerrainBoundary::is_water(int64_t gx, int64_t gy, int64_t gz) const {
    return !is_subsurface(gx, gy, gz);
}

double TerrainBoundary::seafloor_depth(int64_t gx, int64_t gy) const {
    int64_t nx = grid_.params().nx_global;
    int64_t ny = grid_.params().ny_global;
    if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) {
        return water_depth_;
    }
    return bathymetry_grid_[gx * ny + gy];
}

int64_t TerrainBoundary::seafloor_index(int64_t gx, int64_t gy) const {
    double depth = seafloor_depth(gx, gy);
    return static_cast<int64_t>(depth / grid_.params().dz);
}

void TerrainBoundary::apply_free_surface_bc(std::vector<double>& field,
                                             const LocalGrid& loc,
                                             int halo_width) {
    for (int64_t ix = halo_width; ix < loc.nx_with_halo - halo_width; ++ix) {
        for (int64_t iy = halo_width; iy < loc.ny_with_halo - halo_width;
             ++iy) {
            int64_t surface_idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                                  iy * loc.nz_with_halo + halo_width;
            field[surface_idx] = 0.0;
        }
    }
}

void TerrainBoundary::apply_seafloor_bc(std::vector<double>& field,
                                         const LocalGrid& loc,
                                         int halo_width) {
    const auto& params = grid_.params();
    for (int64_t ix = halo_width; ix < loc.nx_with_halo - halo_width; ++ix) {
        for (int64_t iy = halo_width; iy < loc.ny_with_halo - halo_width;
             ++iy) {
            int64_t gx = ix - halo_width + loc.offset_x;
            int64_t gy = iy - halo_width + loc.offset_y;
            int64_t sf_iz = seafloor_index(gx, gy);

            for (int64_t iz = halo_width; iz < halo_width + sf_iz; ++iz) {
                int64_t idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                              iy * loc.nz_with_halo + iz;
                if (idx < static_cast<int64_t>(field.size())) {
                    field[idx] = 0.0;
                }
            }
        }
    }
}

void TerrainBoundary::apply_absorbing_bc(std::vector<double>& field,
                                          const LocalGrid& loc,
                                          int halo_width,
                                          int sponge_width,
                                          double damping) {
    for (int64_t ix = halo_width; ix < loc.nx_with_halo - halo_width; ++ix) {
        for (int64_t iy = halo_width; iy < loc.ny_with_halo - halo_width;
             ++iy) {
            for (int64_t iz = halo_width; iz < loc.nz_with_halo - halo_width;
                 ++iz) {
                int64_t idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                              iy * loc.nz_with_halo + iz;

                double dist_min = std::min({
                    (double)(iz - halo_width),
                    (double)(loc.nz_with_halo - halo_width - 1 - iz),
                    (double)(ix - halo_width),
                    (double)(loc.nx_with_halo - halo_width - 1 - ix),
                    (double)(iy - halo_width),
                    (double)(loc.ny_with_halo - halo_width - 1 - iy)
                });

                if (dist_min < sponge_width && dist_min >= 0) {
                    double factor = 1.0 - damping *
                                        (sponge_width - dist_min) *
                                        (sponge_width - dist_min) /
                                        (sponge_width * sponge_width);
                    field[idx] *= factor;
                }
            }
        }
    }
}

void TerrainBoundary::update_velocity_mask(std::vector<double>& velocity,
                                            double water_vp,
                                            double sediment_vp) const {
    const auto& loc = grid_.local();
    const auto& params = grid_.params();
    int hw = params.halo_width;

    for (int64_t ix = hw; ix < loc.nx_with_halo - hw; ++ix) {
        for (int64_t iy = hw; iy < loc.ny_with_halo - hw; ++iy) {
            int64_t gx = ix - hw + loc.offset_x;
            int64_t gy = iy - hw + loc.offset_y;
            int64_t sf_iz = seafloor_index(gx, gy);

            for (int64_t iz = hw; iz < loc.nz_with_halo - hw; ++iz) {
                int64_t idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                              iy * loc.nz_with_halo + iz;

                if (iz - hw < sf_iz) {
                    velocity[idx] = water_vp;
                } else {
                    double depth_gradient =
                        1.0 + 0.3 * ((iz - hw - sf_iz) * params.dz / 5000.0);
                    velocity[idx] = sediment_vp * depth_gradient;
                    velocity[idx] = std::min(velocity[idx], 8500.0);
                }
            }
        }
    }
}

void TerrainBoundary::build_boundary_mask(const LocalGrid& loc,
                                           int halo_width) {
    int64_t total = loc.nx_with_halo * loc.ny_with_halo * loc.nz_with_halo;
    boundary_mask_.assign(total, 0);

    for (int64_t ix = halo_width; ix < loc.nx_with_halo - halo_width; ++ix) {
        for (int64_t iy = halo_width; iy < loc.ny_with_halo - halo_width;
             ++iy) {
            int64_t gx = ix - halo_width + loc.offset_x;
            int64_t gy = iy - halo_width + loc.offset_y;

            for (int64_t iz = halo_width; iz < loc.nz_with_halo - halo_width;
                 ++iz) {
                int64_t idx = ix * loc.ny_with_halo * loc.nz_with_halo +
                              iy * loc.nz_with_halo + iz;

                if (is_subsurface(gx, gy, iz - halo_width + loc.offset_z)) {
                    boundary_mask_[idx] = 1;
                } else {
                    boundary_mask_[idx] = 2;
                }
            }
        }
    }
}

double TerrainBoundary::bilinear_interp_bathymetry(double x, double y) const {
    int64_t nx = grid_.params().nx_global;
    int64_t ny = grid_.params().ny_global;
    double dx = grid_.params().dx;
    double dy = grid_.params().dy;
    double ox = grid_.params().origin_x;
    double oy = grid_.params().origin_y;

    double fx = (x - ox) / dx;
    double fy = (y - oy) / dy;

    int64_t ix0 = static_cast<int64_t>(std::floor(fx));
    int64_t iy0 = static_cast<int64_t>(std::floor(fy));
    int64_t ix1 = ix0 + 1;
    int64_t iy1 = iy0 + 1;

    ix0 = std::max<int64_t>(0, std::min(ix0, nx - 1));
    ix1 = std::max<int64_t>(0, std::min(ix1, nx - 1));
    iy0 = std::max<int64_t>(0, std::min(iy0, ny - 1));
    iy1 = std::max<int64_t>(0, std::min(iy1, ny - 1));

    double wx = fx - std::floor(fx);
    double wy = fy - std::floor(fy);

    double d00 = bathymetry_grid_[ix0 * ny + iy0];
    double d10 = bathymetry_grid_[ix1 * ny + iy0];
    double d01 = bathymetry_grid_[ix0 * ny + iy1];
    double d11 = bathymetry_grid_[ix1 * ny + iy1];

    return d00 * (1 - wx) * (1 - wy) + d10 * wx * (1 - wy) +
           d01 * (1 - wx) * wy + d11 * wx * wy;
}

}
