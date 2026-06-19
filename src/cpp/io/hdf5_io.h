#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace seismic {

class Hdf5IO {
public:
    static void write_velocity_model(const std::string& filename,
                                     const std::string& dataset_name,
                                     const double* data,
                                     int64_t nx, int64_t ny, int64_t nz,
                                     double dx, double dy, double dz,
                                     int iteration);

    static void read_velocity_model(const std::string& filename,
                                    const std::string& dataset_name,
                                    std::vector<double>& data,
                                    int64_t& nx, int64_t& ny, int64_t& nz);

    static void write_gradient(const std::string& filename,
                               const std::string& dataset_name,
                               const double* data,
                               int64_t nx, int64_t ny, int64_t nz,
                               int iteration);

    static void write_wavefield_snapshot(const std::string& filename,
                                         const std::string& dataset_name,
                                         const double* data,
                                         int64_t nx, int64_t ny, int64_t nz,
                                         int timestep);

    static void write_bathymetry(const std::string& filename,
                                 const double* data,
                                 int64_t nx, int64_t ny,
                                 double dx, double dy);

    static void write_inversion_history(const std::string& filename,
                                        const std::vector<double>& misfits,
                                        const std::vector<int>& iterations);

    static void write_source_receiver_pairs(const std::string& filename,
                                            const std::vector<double>& src_pos,
                                            const std::vector<double>& rcv_pos,
                                            int64_t n_src, int64_t n_rcv);

    static void write_travel_times(const std::string& filename,
                                   const std::vector<double>& times_obs,
                                   const std::vector<double>& times_calc,
                                   const std::vector<int>& src_ids,
                                   const std::vector<int>& rcv_ids);

    static bool file_exists(const std::string& filename);
};

}
