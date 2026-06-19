#include "csr_matrix.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace seismic {

CSRMatrix::CSRMatrix()
    : rows_(0), cols_(0), nnz_(0), block_size_(8) {}

CSRMatrix::CSRMatrix(int64_t rows, int64_t cols, int64_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz), block_size_(8) {
    row_ptr_.resize(rows + 1, 0);
    col_ind_.resize(nnz, 0);
    values_.resize(nnz, 0.0);
}

void CSRMatrix::build_from_triplets(int64_t rows, int64_t cols,
                                     const std::vector<SparseEntry>& entries) {
    rows_ = rows;
    cols_ = cols;

    std::vector<std::vector<std::pair<int64_t, double>>> row_data(rows);

    for (const auto& e : entries) {
        if (e.row >= 0 && e.row < rows && e.col >= 0 && e.col < cols) {
            row_data[e.row].emplace_back(e.col, e.value);
        }
    }

    for (auto& rd : row_data) {
        std::sort(rd.begin(), rd.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<std::pair<int64_t, double>> merged;
        for (const auto& p : rd) {
            if (!merged.empty() && merged.back().first == p.first) {
                merged.back().second += p.second;
            } else {
                merged.push_back(p);
            }
        }
        rd = std::move(merged);
    }

    nnz_ = 0;
    for (const auto& rd : row_data) {
        nnz_ += rd.size();
    }

    row_ptr_.resize(rows + 1);
    col_ind_.resize(nnz_);
    values_.resize(nnz_);

    row_ptr_[0] = 0;
    for (int64_t i = 0; i < rows; ++i) {
        row_ptr_[i + 1] = row_ptr_[i] + static_cast<int64_t>(row_data[i].size());
        for (size_t j = 0; j < row_data[i].size(); ++j) {
            col_ind_[row_ptr_[i] + j] = row_data[i][j].first;
            values_[row_ptr_[i] + j] = row_data[i][j].second;
        }
    }

    pad_to_alignment();
}

void CSRMatrix::build_fd_stencil_7point(int64_t nx, int64_t ny, int64_t nz,
                                         const double* velocity,
                                         double dx, double dy, double dz,
                                         double dt) {
    int64_t n = nx * ny * nz;
    rows_ = n;
    cols_ = n;

    std::vector<SparseEntry> entries;
    entries.reserve(7 * n);

    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double idz2 = 1.0 / (dz * dz);
    double dt2 = dt * dt;

    auto idx = [ny, nz](int64_t ix, int64_t iy, int64_t iz) {
        return ix * ny * nz + iy * nz + iz;
    };

    for (int64_t ix = 0; ix < nx; ++ix) {
        for (int64_t iy = 0; iy < ny; ++iy) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                int64_t r = idx(ix, iy, iz);
                double v = velocity[r];
                double v2dt2 = v * v * dt2;

                double diag = -2.0 * v2dt2 * (idx2 + idy2 + idz2) + 2.0;

                entries.push_back({r, r, diag});

                if (ix > 0)     entries.push_back({r, idx(ix-1, iy, iz), v2dt2 * idx2});
                if (ix < nx-1)  entries.push_back({r, idx(ix+1, iy, iz), v2dt2 * idx2});
                if (iy > 0)     entries.push_back({r, idx(ix, iy-1, iz), v2dt2 * idy2});
                if (iy < ny-1)  entries.push_back({r, idx(ix, iy+1, iz), v2dt2 * idy2});
                if (iz > 0)     entries.push_back({r, idx(ix, iy, iz-1), v2dt2 * idz2});
                if (iz < nz-1)  entries.push_back({r, idx(ix, iy, iz+1), v2dt2 * idz2});
            }
        }
    }

    build_from_triplets(n, n, entries);
}

AlignedVector<double> CSRMatrix::spmv(const AlignedVector<double>& x) const {
    AlignedVector<double> y(rows_, 0.0);

    int64_t rows_per_block = block_size_;
    int64_t num_blocks = (rows_ + rows_per_block - 1) / rows_per_block;

    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t r_start = b * rows_per_block;
        int64_t r_end = std::min(r_start + rows_per_block, rows_);

        for (int64_t i = r_start; i < r_end; ++i) {
            double sum0 = 0.0;
            double sum1 = 0.0;

            int64_t p_start = row_ptr_[i];
            int64_t p_end = row_ptr_[i + 1];
            int64_t p;

            for (p = p_start; p + 1 < p_end; p += 2) {
                sum0 += values_[p] * x[col_ind_[p]];
                sum1 += values_[p + 1] * x[col_ind_[p + 1]];
            }
            for (; p < p_end; ++p) {
                sum0 += values_[p] * x[col_ind_[p]];
            }

            y[i] = sum0 + sum1;
        }
    }

    return y;
}

AlignedVector<double> CSRMatrix::spmv_transpose(
    const AlignedVector<double>& x) const {
    AlignedVector<double> y(cols_, 0.0);

    for (int64_t i = 0; i < rows_; ++i) {
        double xi = x[i];
        int64_t p_start = row_ptr_[i];
        int64_t p_end = row_ptr_[i + 1];

        for (int64_t p = p_start; p < p_end; ++p) {
            y[col_ind_[p]] += values_[p] * xi;
        }
    }

    return y;
}

double CSRMatrix::frobenius_norm() const {
    double sum = 0.0;
    for (int64_t i = 0; i < nnz_; ++i) {
        sum += values_[i] * values_[i];
    }
    return std::sqrt(sum);
}

void CSRMatrix::scale(double alpha) {
    for (int64_t i = 0; i < nnz_; ++i) {
        values_[i] *= alpha;
    }
}

void CSRMatrix::add_diagonal(double alpha) {
    for (int64_t i = 0; i < rows_; ++i) {
        int64_t p_start = row_ptr_[i];
        int64_t p_end = row_ptr_[i + 1];
        for (int64_t p = p_start; p < p_end; ++p) {
            if (col_ind_[p] == i) {
                values_[p] += alpha;
                break;
            }
        }
    }
}

void CSRMatrix::optimize_cache_blocking(int64_t block_size) {
    block_size_ = block_size;
    cache_blocks_.clear();

    int64_t num_blocks = (rows_ + block_size_ - 1) / block_size_;

    for (int64_t b = 0; b < num_blocks; ++b) {
        RowBlock blk;
        blk.row_start = b * block_size_;
        blk.row_end = std::min(blk.row_start + block_size_, rows_);
        blk.nnz_count = row_ptr_[blk.row_end] - row_ptr_[blk.row_start];
        blk.pad_to_cache_line = ((blk.nnz_count * sizeof(double) + CACHE_LINE_SIZE - 1) /
                                  CACHE_LINE_SIZE) * CACHE_LINE_SIZE / sizeof(double);
        cache_blocks_.push_back(blk);
    }
}

void CSRMatrix::prefetch_row(int64_t row) const {
    if (row < 0 || row >= rows_) return;
    int64_t p_start = row_ptr_[row];
    int64_t p_end = row_ptr_[row + 1];

    for (int64_t p = p_start; p < p_end; p += CACHE_LINE_SIZE / sizeof(double)) {
        _mm_prefetch(reinterpret_cast<const char*>(&values_[p]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&col_ind_[p]), _MM_HINT_T0);
    }
}

void CSRMatrix::pad_to_alignment() {
    for (int64_t i = 0; i <= rows_; ++i) {
        int64_t padded = ((row_ptr_[i] + SIMD_ALIGNMENT / sizeof(double) - 1) /
                          (SIMD_ALIGNMENT / sizeof(double))) *
                         (SIMD_ALIGNMENT / sizeof(double));
        if (i < rows_) {
            int64_t current_nnz = row_ptr_[i + 1] - row_ptr_[i];
            if (padded > row_ptr_[i] + current_nnz) {
                int64_t new_nnz = padded - row_ptr_[i];

                AlignedVector<int64_t> new_col_ind(nnz_ + new_nnz - current_nnz);
                AlignedVector<double> new_values(nnz_ + new_nnz - current_nnz);

                int64_t offset = 0;
                for (int64_t r = 0; r < rows_; ++r) {
                    int64_t r_start = row_ptr_[r];
                    int64_t r_nnz = row_ptr_[r + 1] - r_start;
                    int64_t r_padded = ((r_nnz + SIMD_ALIGNMENT / sizeof(double) - 1) /
                                       (SIMD_ALIGNMENT / sizeof(double))) *
                                       (SIMD_ALIGNMENT / sizeof(double));

                    for (int64_t p = 0; p < r_nnz; ++p) {
                        new_col_ind[offset + p] = col_ind_[r_start + p];
                        new_values[offset + p] = values_[r_start + p];
                    }
                    for (int64_t p = r_nnz; p < r_padded; ++p) {
                        new_col_ind[offset + p] = r;
                        new_values[offset + p] = 0.0;
                    }
                    row_ptr_[r] = offset;
                    offset += r_padded;
                }
                row_ptr_[rows_] = offset;

                col_ind_ = std::move(new_col_ind);
                values_ = std::move(new_values);
                nnz_ = offset;
                break;
            }
        }
    }
}

}
