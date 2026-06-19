#pragma once

#include "aligned_alloc.h"
#include <cstdint>
#include <vector>
#include <utility>
#include <functional>

namespace seismic {

struct SparseEntry {
    int64_t row;
    int64_t col;
    double value;
};

class CSRMatrix {
public:
    CSRMatrix();
    CSRMatrix(int64_t rows, int64_t cols, int64_t nnz);

    void build_from_triplets(int64_t rows, int64_t cols,
                             const std::vector<SparseEntry>& entries);

    void build_fd_stencil_7point(int64_t nx, int64_t ny, int64_t nz,
                                 const double* velocity,
                                 double dx, double dy, double dz, double dt);

    AlignedVector<double> spmv(const AlignedVector<double>& x) const;
    AlignedVector<double> spmv_transpose(const AlignedVector<double>& x) const;

    int64_t rows() const { return rows_; }
    int64_t cols() const { return cols_; }
    int64_t nnz() const { return nnz_; }

    const AlignedVector<int64_t>& row_ptr() const { return row_ptr_; }
    const AlignedVector<int64_t>& col_ind() const { return col_ind_; }
    const AlignedVector<double>& values() const { return values_; }

    double frobenius_norm() const;

    void scale(double alpha);

    void add_diagonal(double alpha);

    struct RowBlock {
        int64_t row_start;
        int64_t row_end;
        int64_t nnz_count;
        int64_t pad_to_cache_line;
    };

    void optimize_cache_blocking(int64_t block_size = CACHE_LINE_SIZE / sizeof(double));
    const std::vector<RowBlock>& cache_blocks() const { return cache_blocks_; }

    void prefetch_row(int64_t row) const;

private:
    int64_t rows_;
    int64_t cols_;
    int64_t nnz_;

    AlignedVector<int64_t> row_ptr_;
    AlignedVector<int64_t> col_ind_;
    AlignedVector<double> values_;

    std::vector<RowBlock> cache_blocks_;
    int64_t block_size_;

    void pad_to_alignment();
};

}
