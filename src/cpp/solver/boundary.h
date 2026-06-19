#pragma once

#include "core/grid.h"
#include <vector>
#include <functional>

namespace seismic {

struct BathymetryPoint {
    double x;
    double y;
    double depth;
};

class TerrainBoundary {
public:
    TerrainBoundary(const CartesianGrid& grid, double water_depth);

    void set_bathymetry_data(const std::vector<BathymetryPoint>& data);
    void generate_synthetic_bathymetry(double gradient, double roughness);

    bool is_subsurface(int64_t gx, int64_t gy, int64_t gz) const;
    bool is_water(int64_t gx, int64_t gy, int64_t gz) const;

    double seafloor_depth(int64_t gx, int64_t gy) const;
    int64_t seafloor_index(int64_t gx, int64_t gy) const;

    void apply_free_surface_bc(std::vector<double>& field,
                               const LocalGrid& loc, int halo_width);
    void apply_seafloor_bc(std::vector<double>& field,
                           const LocalGrid& loc, int halo_width);
    void apply_absorbing_bc(std::vector<double>& field,
                            const LocalGrid& loc, int halo_width,
                            int sponge_width, double damping);

    void update_velocity_mask(std::vector<double>& velocity,
                              double water_vp, double sediment_vp) const;

    const std::vector<double>& bathymetry() const { return bathymetry_grid_; }

    void build_boundary_mask(const LocalGrid& loc, int halo_width);

    const std::vector<uint8_t>& mask() const { return boundary_mask_; }

private:
    const CartesianGrid& grid_;
    double water_depth_;
    std::vector<double> bathymetry_grid_;
    std::vector<uint8_t> boundary_mask_;

    double bilinear_interp_bathymetry(double x, double y) const;
};

}
