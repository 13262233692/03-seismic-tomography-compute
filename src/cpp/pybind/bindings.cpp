#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sparse/csr_matrix.h"
#include "sparse/sparse_ops.h"
#include "solver/boundary.h"
#include "io/hdf5_io.h"

namespace py = pybind11;

PYBIND11_MODULE(seismic_engine, m) {
    m.doc() = "Deep-sea seismic wave tomography C++ engine with MPI parallelism";

    py::class_<seismic::CSRMatrix>(m, "CSRMatrix")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>())
        .def("build_from_triplets", [](seismic::CSRMatrix& self,
                                        int64_t rows, int64_t cols,
                                        py::array_t<int64_t> row_idx,
                                        py::array_t<int64_t> col_idx,
                                        py::array_t<double> values) {
            auto r = row_idx.unchecked<1>();
            auto c = col_idx.unchecked<1>();
            auto v = values.unchecked<1>();
            std::vector<seismic::SparseEntry> entries;
            entries.reserve(r.shape(0));
            for (ssize_t i = 0; i < r.shape(0); ++i) {
                entries.push_back({r(i), c(i), v(i)});
            }
            self.build_from_triplets(rows, cols, entries);
        }, py::arg("rows"), py::arg("cols"),
           py::arg("row_idx"), py::arg("col_idx"), py::arg("values"))
        .def("build_fd_stencil_7point", [](seismic::CSRMatrix& self,
                                            int64_t nx, int64_t ny, int64_t nz,
                                            py::array_t<double, py::array::c_style | py::array::forcecast> velocity,
                                            double dx, double dy, double dz, double dt) {
            auto v = velocity.unchecked<1>();
            self.build_fd_stencil_7point(nx, ny, nz, v.data(0), dx, dy, dz, dt);
        }, py::arg("nx"), py::arg("ny"), py::arg("nz"),
           py::arg("velocity"), py::arg("dx"), py::arg("dy"),
           py::arg("dz"), py::arg("dt"))
        .def("spmv", [](seismic::CSRMatrix& self,
                         py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            auto x_buf = x.unchecked<1>();
            seismic::AlignedVector<double> x_vec(x_buf.shape(0));
            for (ssize_t i = 0; i < x_buf.shape(0); ++i) {
                x_vec[i] = x_buf(i);
            }
            seismic::AlignedVector<double> y = self.spmv(x_vec);
            auto result = py::array_t<double>(y.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < y.size(); ++i) {
                r(i) = y[i];
            }
            return result;
        }, py::arg("x"))
        .def("spmv_transpose", [](seismic::CSRMatrix& self,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            auto x_buf = x.unchecked<1>();
            seismic::AlignedVector<double> x_vec(x_buf.shape(0));
            for (ssize_t i = 0; i < x_buf.shape(0); ++i) {
                x_vec[i] = x_buf(i);
            }
            seismic::AlignedVector<double> y = self.spmv_transpose(x_vec);
            auto result = py::array_t<double>(y.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < y.size(); ++i) {
                r(i) = y[i];
            }
            return result;
        }, py::arg("x"))
        .def("frobenius_norm", &seismic::CSRMatrix::frobenius_norm)
        .def("scale", &seismic::CSRMatrix::scale, py::arg("alpha"))
        .def("add_diagonal", &seismic::CSRMatrix::add_diagonal, py::arg("alpha"))
        .def("optimize_cache_blocking", &seismic::CSRMatrix::optimize_cache_blocking,
             py::arg("block_size") = 8)
        .def("rows", &seismic::CSRMatrix::rows)
        .def("cols", &seismic::CSRMatrix::cols)
        .def("nnz", &seismic::CSRMatrix::nnz)
        .def("row_ptr", [](seismic::CSRMatrix& self) {
            auto& rp = self.row_ptr();
            auto result = py::array_t<int64_t>(rp.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < rp.size(); ++i) r(i) = rp[i];
            return result;
        })
        .def("col_ind", [](seismic::CSRMatrix& self) {
            auto& ci = self.col_ind();
            auto result = py::array_t<int64_t>(ci.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < ci.size(); ++i) r(i) = ci[i];
            return result;
        })
        .def("values", [](seismic::CSRMatrix& self) {
            auto& v = self.values();
            auto result = py::array_t<double>(v.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < v.size(); ++i) r(i) = v[i];
            return result;
        });

    py::class_<seismic::SparseOps>(m, "SparseOps")
        .def_static("conjugate_gradient", [](seismic::CSRMatrix& A,
                                              py::array_t<double, py::array::c_style | py::array::forcecast> b,
                                              int max_iter, double tolerance) {
            auto b_buf = b.unchecked<1>();
            seismic::AlignedVector<double> b_vec(b_buf.shape(0));
            for (ssize_t i = 0; i < b_buf.shape(0); ++i) b_vec[i] = b_buf(i);
            auto x = seismic::SparseOps::conjugate_gradient(A, b_vec, max_iter, tolerance);
            auto result = py::array_t<double>(x.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < x.size(); ++i) r(i) = x[i];
            return result;
        }, py::arg("A"), py::arg("b"),
           py::arg("max_iter") = 1000, py::arg("tolerance") = 1e-8)
        .def_static("jacobi_preconditioned_cg", [](seismic::CSRMatrix& A,
                                                     py::array_t<double, py::array::c_style | py::array::forcecast> b,
                                                     int max_iter, double tolerance) {
            auto b_buf = b.unchecked<1>();
            seismic::AlignedVector<double> b_vec(b_buf.shape(0));
            for (ssize_t i = 0; i < b_buf.shape(0); ++i) b_vec[i] = b_buf(i);
            auto x = seismic::SparseOps::jacobi_preconditioned_cg(A, b_vec, max_iter, tolerance);
            auto result = py::array_t<double>(x.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < x.size(); ++i) r(i) = x[i];
            return result;
        }, py::arg("A"), py::arg("b"),
           py::arg("max_iter") = 1000, py::arg("tolerance") = 1e-8)
        .def_static("smooth_gradient", [](py::array_t<double, py::array::c_style | py::array::forcecast> gradient,
                                           int64_t nx, int64_t ny, int64_t nz,
                                           double sigma) {
            auto g = gradient.unchecked<1>();
            std::vector<double> grad_vec(g.shape(0));
            for (ssize_t i = 0; i < g.shape(0); ++i) grad_vec[i] = g(i);
            auto smoothed = seismic::SparseOps::smooth_gradient(grad_vec, nx, ny, nz, sigma);
            auto result = py::array_t<double>(smoothed.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < smoothed.size(); ++i) r(i) = smoothed[i];
            return result;
        }, py::arg("gradient"), py::arg("nx"), py::arg("ny"),
           py::arg("nz"), py::arg("sigma"))
        .def_static("dot_product", [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> b) {
            auto a_buf = a.unchecked<1>();
            auto b_buf = b.unchecked<1>();
            seismic::AlignedVector<double> a_vec(a_buf.shape(0));
            seismic::AlignedVector<double> b_vec(b_buf.shape(0));
            for (ssize_t i = 0; i < a_buf.shape(0); ++i) a_vec[i] = a_buf(i);
            for (ssize_t i = 0; i < b_buf.shape(0); ++i) b_vec[i] = b_buf(i);
            return seismic::SparseOps::dot_product(a_vec, b_vec);
        }, py::arg("a"), py::arg("b"))
        .def_static("build_hessian_approx", [](seismic::CSRMatrix& forward_op,
                                                double regularization) {
            return seismic::SparseOps::build_hessian_approx(forward_op, regularization);
        }, py::arg("forward_operator"), py::arg("regularization"));

    py::class_<seismic::BathymetryPoint>(m, "BathymetryPoint")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &seismic::BathymetryPoint::x)
        .def_readwrite("y", &seismic::BathymetryPoint::y)
        .def_readwrite("depth", &seismic::BathymetryPoint::depth);

    py::class_<seismic::Hdf5IO>(m, "Hdf5IO")
        .def_static("write_velocity_model", [](const std::string& filename,
                                                const std::string& dataset_name,
                                                py::array_t<double, py::array::c_style | py::array::forcecast> data,
                                                int64_t nx, int64_t ny, int64_t nz,
                                                double dx, double dy, double dz,
                                                int iteration) {
            auto d = data.unchecked<1>();
            seismic::Hdf5IO::write_velocity_model(filename, dataset_name,
                                                   d.data(0), nx, ny, nz,
                                                   dx, dy, dz, iteration);
        }, py::arg("filename"), py::arg("dataset_name"), py::arg("data"),
           py::arg("nx"), py::arg("ny"), py::arg("nz"),
           py::arg("dx"), py::arg("dy"), py::arg("dz"), py::arg("iteration"))
        .def_static("read_velocity_model", [](const std::string& filename,
                                               const std::string& dataset_name) {
            std::vector<double> data;
            int64_t nx, ny, nz;
            seismic::Hdf5IO::read_velocity_model(filename, dataset_name, data, nx, ny, nz);
            auto result = py::array_t<double>(data.size());
            auto r = result.mutable_unchecked<1>();
            for (size_t i = 0; i < data.size(); ++i) r(i) = data[i];
            py::tuple shape(3);
            shape[0] = nx;
            shape[1] = ny;
            shape[2] = nz;
            return py::make_tuple(result.reshape(shape), shape);
        }, py::arg("filename"), py::arg("dataset_name"))
        .def_static("write_bathymetry", [](const std::string& filename,
                                            py::array_t<double, py::array::c_style | py::array::forcecast> data,
                                            int64_t nx, int64_t ny,
                                            double dx, double dy) {
            auto d = data.unchecked<1>();
            seismic::Hdf5IO::write_bathymetry(filename, d.data(0), nx, ny, dx, dy);
        })
        .def_static("write_inversion_history", [](const std::string& filename,
                                                    std::vector<double> misfits,
                                                    std::vector<int> iterations) {
            seismic::Hdf5IO::write_inversion_history(filename, misfits, iterations);
        });

    m.def("compute_fd_stencil", [](int64_t nx, int64_t ny, int64_t nz,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> velocity,
                                    double dx, double dy, double dz, double dt) {
        auto v = velocity.unchecked<1>();
        seismic::CSRMatrix mat;
        mat.build_fd_stencil_7point(nx, ny, nz, v.data(0), dx, dy, dz, dt);
        return mat;
    }, py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("velocity"), py::arg("dx"), py::arg("dy"),
       py::arg("dz"), py::arg("dt"),
       "Build the 7-point finite difference stencil as a sparse CSR matrix");

    m.def("generate_bathymetry", [](int64_t nx, int64_t ny,
                                     double dx, double dy,
                                     double origin_x, double origin_y,
                                     double water_depth,
                                     double gradient, double roughness) {
        seismic::GridParams params{nx, ny, 1, dx, dy, 1.0, origin_x, origin_y, 0.0, 0};
        seismic::CartesianGrid grid(params, (int[]){0, 0, 0}, (int[]){1, 1, 1});
        seismic::TerrainBoundary boundary(grid, water_depth);
        boundary.generate_synthetic_bathymetry(gradient, roughness);
        auto& bath = boundary.bathymetry();

        auto result = py::array_t<double>(nx * ny);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < bath.size(); ++i) r(i) = bath[i];
        return result.reshape({static_cast<ssize_t>(nx), static_cast<ssize_t>(ny)});
    }, py::arg("nx"), py::arg("ny"), py::arg("dx"), py::arg("dy"),
       py::arg("origin_x"), py::arg("origin_y"), py::arg("water_depth"),
       py::arg("gradient"), py::arg("roughness"),
       "Generate synthetic deep-sea bathymetry for irregular terrain boundary conditions");
}
