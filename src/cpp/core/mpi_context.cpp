#include "mpi_context.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace seismic {

MpiContext::MpiContext(int& argc, char**& argv) : owns_mpi_(false) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int finalized;
    MPI_Finalized(&finalized);
    owns_mpi_ = !finalized;

    MPI_Comm_rank(MPI_COMM_WORLD, &cfg_.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg_.size);

    cfg_.dims[0] = 0;
    cfg_.dims[1] = 0;
    cfg_.dims[2] = 0;

    MPI_Dims_create(cfg_.size, 3, cfg_.dims);

    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, cfg_.dims, periods, 1, &cfg_.cart_comm);

    MPI_Comm_rank(cfg_.cart_comm, &cfg_.rank);
    MPI_Cart_coords(cfg_.cart_comm, cfg_.rank, 3, cfg_.coords);

    setup_neighbors();
}

MpiContext::~MpiContext() {
    if (owns_mpi_) {
        MPI_Comm_free(&cfg_.cart_comm);
        MPI_Finalize();
    }
}

void MpiContext::setup_neighbors() {
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cfg_.cart_comm, d, 1,
                       &cfg_.neighbors[2 * d],
                       &cfg_.neighbors[2 * d + 1]);
    }
}

void MpiContext::barrier() const {
    MPI_Barrier(cfg_.cart_comm);
}

void MpiContext::abort(int error_code) const {
    MPI_Abort(cfg_.cart_comm, error_code);
}

double MpiContext::allreduce_max(double local_val) const {
    double global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, cfg_.cart_comm);
    return global_val;
}

double MpiContext::allreduce_sum(double local_val) const {
    double global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, cfg_.cart_comm);
    return global_val;
}

void MpiContext::send_double(const double* buf, int count, int dest, int tag) const {
    MPI_Send(buf, count, MPI_DOUBLE, dest, tag, cfg_.cart_comm);
}

void MpiContext::recv_double(double* buf, int count, int src, int tag) const {
    MPI_Recv(buf, count, MPI_DOUBLE, src, tag, cfg_.cart_comm, MPI_STATUS_IGNORE);
}

void MpiContext::allgather_int(const int* sendbuf, int sendcount,
                               int* recvbuf, int recvcount) const {
    MPI_Allgather(sendbuf, sendcount, MPI_INT,
                  recvbuf, recvcount, MPI_INT, cfg_.cart_comm);
}

void MpiContext::bcast_double(double* buf, int count, int root) const {
    MPI_Bcast(buf, count, MPI_DOUBLE, root, cfg_.cart_comm);
}

void MpiContext::bcast_int(int* buf, int count, int root) const {
    MPI_Bcast(buf, count, MPI_INT, root, cfg_.cart_comm);
}

double MpiContext::time() const {
    return MPI_Wtime();
}

void MpiContext::print_rank_info() const {
    for (int r = 0; r < cfg_.size; ++r) {
        if (r == cfg_.rank) {
            printf("[Rank %d] coords=(%d,%d,%d) dims=(%d,%d,%d) neighbors="
                   "[%d,%d,%d,%d,%d,%d]\n",
                   cfg_.rank,
                   cfg_.coords[0], cfg_.coords[1], cfg_.coords[2],
                   cfg_.dims[0], cfg_.dims[1], cfg_.dims[2],
                   cfg_.neighbors[0], cfg_.neighbors[1],
                   cfg_.neighbors[2], cfg_.neighbors[3],
                   cfg_.neighbors[4], cfg_.neighbors[5]);
            fflush(stdout);
        }
        MPI_Barrier(cfg_.cart_comm);
    }
}

}
