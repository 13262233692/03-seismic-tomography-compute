#include "hdf5_io.h"
#include <H5Cpp.h>
#include <vector>
#include <string>
#include <fstream>

namespace seismic {

bool Hdf5IO::file_exists(const std::string& filename) {
    std::ifstream ifs(filename);
    return ifs.good();
}

void Hdf5IO::write_velocity_model(const std::string& filename,
                                   const std::string& dataset_name,
                                   const double* data,
                                   int64_t nx, int64_t ny, int64_t nz,
                                   double dx, double dy, double dz,
                                   int iteration) {
    H5::H5File* file;
    if (file_exists(filename)) {
        file = new H5::H5File(filename, H5F_ACC_RDWR);
    } else {
        file = new H5::H5File(filename, H5F_ACC_TRUNC);
    }

    std::string dname = dataset_name + "/iter_" + std::to_string(iteration);

    hsize_t dims[3] = {static_cast<hsize_t>(nx),
                       static_cast<hsize_t>(ny),
                       static_cast<hsize_t>(nz)};
    H5::DataSpace dataspace(3, dims);

    H5::FloatType datatype(H5::PredType::NATIVE_DOUBLE);
    H5::DSetCreatPropList plist;
    plist.setChunk(3, dims);
    plist.setDeflate(6);

    H5::DataSet dataset = file->createDataSet(dname, datatype, dataspace, plist);
    dataset.write(data, H5::PredType::NATIVE_DOUBLE);

    H5::Attribute attr_dx = dataset.createAttribute(
        "dx", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
    attr_dx.write(H5::PredType::NATIVE_DOUBLE, &dx);

    H5::Attribute attr_dy = dataset.createAttribute(
        "dy", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
    attr_dy.write(H5::PredType::NATIVE_DOUBLE, &dy);

    H5::Attribute attr_dz = dataset.createAttribute(
        "dz", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
    attr_dz.write(H5::PredType::NATIVE_DOUBLE, &dz);

    H5::Attribute attr_iter = dataset.createAttribute(
        "iteration", H5::PredType::NATIVE_INT32, H5::DataSpace(H5S_SCALAR));
    attr_iter.write(H5::PredType::NATIVE_INT, &iteration);

    delete file;
}

void Hdf5IO::read_velocity_model(const std::string& filename,
                                  const std::string& dataset_name,
                                  std::vector<double>& data,
                                  int64_t& nx, int64_t& ny, int64_t& nz) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims);
    nx = dims[0];
    ny = dims[1];
    nz = dims[2];

    data.resize(nx * ny * nz);
    dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);
}

void Hdf5IO::write_gradient(const std::string& filename,
                             const std::string& dataset_name,
                             const double* data,
                             int64_t nx, int64_t ny, int64_t nz,
                             int iteration) {
    write_velocity_model(filename, dataset_name, data,
                         nx, ny, nz, 0, 0, 0, iteration);
}

void Hdf5IO::write_wavefield_snapshot(const std::string& filename,
                                       const std::string& dataset_name,
                                       const double* data,
                                       int64_t nx, int64_t ny, int64_t nz,
                                       int timestep) {
    H5::H5File* file;
    if (file_exists(filename)) {
        file = new H5::H5File(filename, H5F_ACC_RDWR);
    } else {
        file = new H5::H5File(filename, H5F_ACC_TRUNC);
    }

    std::string dname = dataset_name + "/step_" + std::to_string(timestep);

    hsize_t dims[3] = {static_cast<hsize_t>(nx),
                       static_cast<hsize_t>(ny),
                       static_cast<hsize_t>(nz)};
    H5::DataSpace dataspace(3, dims);

    H5::DSetCreatPropList plist;
    plist.setChunk(3, dims);
    plist.setDeflate(4);

    H5::DataSet dataset = file->createDataSet(
        dname, H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    dataset.write(data, H5::PredType::NATIVE_DOUBLE);

    delete file;
}

void Hdf5IO::write_bathymetry(const std::string& filename,
                               const double* data,
                               int64_t nx, int64_t ny,
                               double dx, double dy) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    hsize_t dims[2] = {static_cast<hsize_t>(nx), static_cast<hsize_t>(ny)};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = file.createDataSet(
        "bathymetry", H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(data, H5::PredType::NATIVE_DOUBLE);

    H5::Attribute attr_dx = dataset.createAttribute(
        "dx", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
    attr_dx.write(H5::PredType::NATIVE_DOUBLE, &dx);

    H5::Attribute attr_dy = dataset.createAttribute(
        "dy", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
    attr_dy.write(H5::PredType::NATIVE_DOUBLE, &dy);
}

void Hdf5IO::write_inversion_history(const std::string& filename,
                                      const std::vector<double>& misfits,
                                      const std::vector<int>& iterations) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    hsize_t n = misfits.size();
    H5::DataSpace dataspace(1, &n);
    H5::DataSet ds_misfit = file.createDataSet(
        "misfit_history", H5::PredType::NATIVE_DOUBLE, dataspace);
    ds_misfit.write(misfits.data(), H5::PredType::NATIVE_DOUBLE);

    H5::DataSet ds_iter = file.createDataSet(
        "iteration_history", H5::PredType::NATIVE_INT32, dataspace);
    ds_iter.write(iterations.data(), H5::PredType::NATIVE_INT);
}

void Hdf5IO::write_source_receiver_pairs(
    const std::string& filename,
    const std::vector<double>& src_pos,
    const std::vector<double>& rcv_pos,
    int64_t n_src, int64_t n_rcv) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    hsize_t src_dims[2] = {static_cast<hsize_t>(n_src), 3};
    H5::DataSpace src_space(2, src_dims);
    H5::DataSet ds_src = file.createDataSet(
        "source_positions", H5::PredType::NATIVE_DOUBLE, src_space);
    ds_src.write(src_pos.data(), H5::PredType::NATIVE_DOUBLE);

    hsize_t rcv_dims[2] = {static_cast<hsize_t>(n_rcv), 3};
    H5::DataSpace rcv_space(2, rcv_dims);
    H5::DataSet ds_rcv = file.createDataSet(
        "receiver_positions", H5::PredType::NATIVE_DOUBLE, rcv_space);
    ds_rcv.write(rcv_pos.data(), H5::PredType::NATIVE_DOUBLE);
}

void Hdf5IO::write_travel_times(const std::string& filename,
                                 const std::vector<double>& times_obs,
                                 const std::vector<double>& times_calc,
                                 const std::vector<int>& src_ids,
                                 const std::vector<int>& rcv_ids) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    hsize_t n = times_obs.size();
    H5::DataSpace dataspace(1, &n);

    H5::DataSet ds_obs = file.createDataSet(
        "observed_travel_times", H5::PredType::NATIVE_DOUBLE, dataspace);
    ds_obs.write(times_obs.data(), H5::PredType::NATIVE_DOUBLE);

    H5::DataSet ds_calc = file.createDataSet(
        "calculated_travel_times", H5::PredType::NATIVE_DOUBLE, dataspace);
    ds_calc.write(times_calc.data(), H5::PredType::NATIVE_DOUBLE);

    H5::DataSet ds_src = file.createDataSet(
        "source_ids", H5::PredType::NATIVE_INT32, dataspace);
    ds_src.write(src_ids.data(), H5::PredType::NATIVE_INT);

    H5::DataSet ds_rcv = file.createDataSet(
        "receiver_ids", H5::PredType::NATIVE_INT32, dataspace);
    ds_rcv.write(rcv_ids.data(), H5::PredType::NATIVE_INT);
}

}
