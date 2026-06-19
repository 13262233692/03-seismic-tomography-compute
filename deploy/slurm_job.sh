#!/bin/bash
#SBATCH --job-name=seismic_tomo
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/seismic_%j.out
#SBATCH --error=logs/seismic_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@institution.edu

module load mpi/openmpi-x86_64
module load hdf5/1.12.0
module load python/3.9
module load cmake/3.22

export OMP_NUM_THREADS=1
export MPI_THREADS_PER_PROC=1
export KMP_AFFINITY=compact

BUILD_DIR=${SLURM_SUBMIT_DIR}/build
if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_MPI=ON \
        -DENABLE_HDF5=ON \
        -DENABLE_PYBIND=ON \
        -DENABLE_AVX2=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math"
fi

cd ${BUILD_DIR}
make -j${SLURM_NTASKS_PER_NODE}

TOTAL_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

NX=200
NY=200
NZ=100
DX=500.0
DY=500.0
DZ=200.0
DT=0.001
MAX_ITER=50

mpirun -np ${TOTAL_TASKS} \
    --bind-to core \
    --map-by core \
    python3 ${SLURM_SUBMIT_DIR}/src/python/scripts/run_inversion.py \
    --nx ${NX} --ny ${NY} --nz ${NZ} \
    --dx ${DX} --dy ${DY} --dz ${DZ} \
    --dt ${DT} \
    --max-iter ${MAX_ITER} \
    --output-dir ${SLURM_SUBMIT_DIR}/output/run_${SLURM_JOB_ID} \
    --log-level INFO

echo "Job ${SLURM_JOB_ID} completed at $(date)"
