#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <cstdint>

namespace seismic {

struct MpiConfig {
    int rank;
    int size;
    int local_rank;
    MPI_Comm cart_comm;
    int dims[3];
    int coords[3];
    int neighbors[6];
};

class MpiContext {
public:
    MpiContext(int& argc, char**& argv);
    ~MpiContext();

    MpiContext(const MpiContext&) = delete;
    MpiContext& operator=(const MpiContext&) = delete;

    const MpiConfig& config() const { return cfg_; }

    void barrier() const;
    void abort(int error_code) const;

    double allreduce_max(double local_val) const;
    double allreduce_sum(double local_val) const;

    void send_double(const double* buf, int count, int dest, int tag) const;
    void recv_double(double* buf, int count, int src, int tag) const;

    void allgather_int(const int* sendbuf, int sendcount,
                       int* recvbuf, int recvcount) const;

    void bcast_double(double* buf, int count, int root) const;
    void bcast_int(int* buf, int count, int root) const;

    double time() const;

    void print_rank_info() const;

private:
    MpiConfig cfg_;
    bool owns_mpi_;

    void setup_cartesian_topology(int nx, int ny, int nz);
    void setup_neighbors();
};

}
