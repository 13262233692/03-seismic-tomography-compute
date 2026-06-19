#include "sparse_ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace seismic {

AlignedVector<double> SparseOps::conjugate_gradient(
    const CSRMatrix& A,
    const AlignedVector<double>& b,
    int max_iter,
    double tolerance) {

    int64_t n = A.rows();
    AlignedVector<double> x(n, 0.0);

    AlignedVector<double> r = b;
    AlignedVector<double> p = r;
    double rs_old = dot_product(r, r);
    double b_norm = l2_norm(b);

    if (b_norm < 1e-30) return x;

    for (int iter = 0; iter < max_iter; ++iter) {
        AlignedVector<double> Ap = A.spmv(p);

        double pAp = dot_product(p, Ap);
        if (std::abs(pAp) < 1e-30) break;

        double alpha = rs_old / pAp;

        for (int64_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rs_new = dot_product(r, r);

        if (std::sqrt(rs_new) / b_norm < tolerance) break;

        double beta = rs_new / rs_old;
        for (int64_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    return x;
}

AlignedVector<double> SparseOps::jacobi_preconditioned_cg(
    const CSRMatrix& A,
    const AlignedVector<double>& b,
    int max_iter,
    double tolerance) {

    int64_t n = A.rows();
    AlignedVector<double> x(n, 0.0);
    AlignedVector<double> diag_inv(n, 1.0);

    for (int64_t i = 0; i < n; ++i) {
        int64_t p_start = A.row_ptr()[i];
        int64_t p_end = A.row_ptr()[i + 1];
        for (int64_t p = p_start; p < p_end; ++p) {
            if (A.col_ind()[p] == i) {
                if (std::abs(A.values()[p]) > 1e-30) {
                    diag_inv[i] = 1.0 / A.values()[p];
                }
                break;
            }
        }
    }

    AlignedVector<double> r = b;
    AlignedVector<double> z(n);
    for (int64_t i = 0; i < n; ++i) z[i] = diag_inv[i] * r[i];
    AlignedVector<double> p = z;

    double rz_old = dot_product(r, z);
    double b_norm = l2_norm(b);

    if (b_norm < 1e-30) return x;

    for (int iter = 0; iter < max_iter; ++iter) {
        AlignedVector<double> Ap = A.spmv(p);
        double pAp = dot_product(p, Ap);
        if (std::abs(pAp) < 1e-30) break;

        double alpha = rz_old / pAp;

        for (int64_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double r_norm = l2_norm(r);
        if (r_norm / b_norm < tolerance) break;

        for (int64_t i = 0; i < n; ++i) z[i] = diag_inv[i] * r[i];

        double rz_new = dot_product(r, z);
        double beta = rz_new / rz_old;

        for (int64_t i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    return x;
}

std::vector<double> SparseOps::smooth_gradient(
    const std::vector<double>& gradient,
    int64_t nx, int64_t ny, int64_t nz,
    double sigma) {
    auto kernel = gaussian_filter_3d(nx, ny, nz, sigma);
    int64_t n = nx * ny * nz;
    std::vector<double> result(n, 0.0);

    if (gradient.size() < static_cast<size_t>(n)) return result;

    auto idx = [ny, nz](int64_t ix, int64_t iy, int64_t iz) {
        return ix * ny * nz + iy * nz + iz;
    };

    int64_t radius = static_cast<int64_t>(3 * sigma);
    for (int64_t ix = 0; ix < nx; ++ix) {
        for (int64_t iy = 0; iy < ny; ++iy) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                double sum = 0.0;
                double wsum = 0.0;
                for (int64_t dx = -radius; dx <= radius; ++dx) {
                    for (int64_t dy = -radius; dy <= radius; ++dy) {
                        for (int64_t dz = -radius; dz <= radius; ++dz) {
                            int64_t jx = ix + dx;
                            int64_t jy = iy + dy;
                            int64_t jz = iz + dz;
                            if (jx >= 0 && jx < nx &&
                                jy >= 0 && jy < ny &&
                                jz >= 0 && jz < nz) {
                                double w = std::exp(-(dx*dx + dy*dy + dz*dz) /
                                                    (2.0 * sigma * sigma));
                                sum += w * gradient[idx(jx, jy, jz)];
                                wsum += w;
                            }
                        }
                    }
                }
                result[idx(ix, iy, iz)] = sum / wsum;
            }
        }
    }

    return result;
}

std::vector<double> SparseOps::compute_directional_derivative(
    const CSRMatrix& A,
    const AlignedVector<double>& direction,
    const AlignedVector<double>& gradient) {
    AlignedVector<double> Ad = A.spmv(direction);
    std::vector<double> result(gradient.size());
    for (int64_t i = 0; i < static_cast<int64_t>(gradient.size()); ++i) {
        result[i] = Ad[i];
    }
    return result;
}

double SparseOps::dot_product(const AlignedVector<double>& a,
                               const AlignedVector<double>& b) {
    double sum = 0.0;
    int64_t n = std::min(a.size(), b.size());

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        sum += a[i] * b[i] + a[i+1] * b[i+1] +
               a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

double SparseOps::l2_norm(const AlignedVector<double>& a) {
    return std::sqrt(dot_product(a, a));
}

AlignedVector<double> SparseOps::axpy(double alpha,
                                       const AlignedVector<double>& x,
                                       const AlignedVector<double>& y) {
    int64_t n = std::min(x.size(), y.size());
    AlignedVector<double> result(n);
    for (int64_t i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

CSRMatrix SparseOps::build_hessian_approx(
    const CSRMatrix& forward_operator,
    double regularization) {
    CSRMatrix H = forward_operator;
    H.add_diagonal(regularization);
    return H;
}

std::vector<double> SparseOps::gaussian_filter_3d(
    int64_t nx, int64_t ny, int64_t nz, double sigma) {
    int64_t n = nx * ny * nz;
    std::vector<double> kernel(n, 0.0);
    double sum = 0.0;
    double s2 = 2.0 * sigma * sigma;

    for (int64_t ix = 0; ix < nx; ++ix) {
        for (int64_t iy = 0; iy < ny; ++iy) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                double cx = ix - nx / 2.0;
                double cy = iy - ny / 2.0;
                double cz = iz - nz / 2.0;
                double val = std::exp(-(cx*cx + cy*cy + cz*cz) / s2);
                kernel[ix * ny * nz + iy * nz + iz] = val;
                sum += val;
            }
        }
    }

    for (auto& v : kernel) v /= sum;
    return kernel;
}

}
