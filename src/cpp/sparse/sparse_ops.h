#pragma once

#include "csr_matrix.h"
#include <vector>
#include <tuple>

namespace seismic {

class SparseOps {
public:
    static AlignedVector<double> conjugate_gradient(
        const CSRMatrix& A,
        const AlignedVector<double>& b,
        int max_iter,
        double tolerance);

    static AlignedVector<double> jacobi_preconditioned_cg(
        const CSRMatrix& A,
        const AlignedVector<double>& b,
        int max_iter,
        double tolerance);

    static std::vector<double> smooth_gradient(
        const std::vector<double>& gradient,
        int64_t nx, int64_t ny, int64_t nz,
        double sigma);

    static std::vector<double> compute_directional_derivative(
        const CSRMatrix& A,
        const AlignedVector<double>& direction,
        const AlignedVector<double>& gradient);

    static double dot_product(const AlignedVector<double>& a,
                              const AlignedVector<double>& b);

    static double l2_norm(const AlignedVector<double>& a);

    static AlignedVector<double> axpy(double alpha,
                                       const AlignedVector<double>& x,
                                       const AlignedVector<double>& y);

    static CSRMatrix build_hessian_approx(
        const CSRMatrix& forward_operator,
        double regularization);

    static std::vector<double> gaussian_filter_3d(
        int64_t nx, int64_t ny, int64_t nz, double sigma);
};

}
