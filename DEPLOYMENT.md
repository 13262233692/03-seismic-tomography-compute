# 深海地震波层析成像系统 — 编译与超算集群部署指南

## 1. 系统架构概览

本系统采用 **C++ / Python 混合架构**，跨进程高效协同：

| 层级 | 技术栈 | 职责 |
|------|--------|------|
| **C++ 引擎层** | MPI + 有限差分 + AVX2 SIMD | 三维波动方程并行求解、稀疏矩阵运算、内存对齐优化 |
| **绑定层** | pybind11 | 将 CSR 稀疏矩阵运算节点暴露给 Python |
| **Python 协调层** | NumPy + SciPy + h5py | OBS 走时数据管理、伴随状态法反演迭代、HDF5 结果输出 |
| **存储层** | HDF5 (压缩分块) | 速度模型、梯度、波形快照、反演历史 |

### 目录结构

```
03-seismic-tomography-compute/
├── CMakeLists.txt                    # CMake 主构建文件
├── config/
│   └── default.yaml                  # 默认参数配置
├── deploy/
│   └── slurm_job.sh                  # SLURM 作业提交脚本
├── src/
│   ├── cpp/
│   │   ├── core/
│   │   │   ├── mpi_context.h/cpp     # MPI 笛卡尔拓扑通信
│   │   │   ├── grid.h/cpp            # 三维网格域分解
│   │   │   └── config.h/cpp          # YAML 参数加载
│   │   ├── solver/
│   │   │   ├── wave_equation.h/cpp   # 7点有限差分波动方程求解器
│   │   │   └── boundary.h/cpp        # 不规则海底地形边界条件
│   │   ├── sparse/
│   │   │   ├── aligned_alloc.h       # 64字节对齐内存分配器
│   │   │   ├── csr_matrix.h/cpp      # SIMD 对齐 CSR 稀疏矩阵
│   │   │   └── sparse_ops.h/cpp      # CG/Jacobi-PCG 求解器
│   │   ├── io/
│   │   │   └── hdf5_io.h/cpp         # HDF5 读写接口
│   │   └── pybind/
│   │       └── bindings.cpp          # pybind11 绑定
│   └── python/
│       ├── seismic_tomography/
│       │   ├── __init__.py
│       │   ├── inversion.py          # 伴随状态法反演主控
│       │   ├── obs_data.py           # OBS 数据协调器
│       │   ├── gradient.py           # 梯度计算与预处理
│       │   ├── model.py              # 速度模型管理
│       │   └── hdf5_writer.py        # HDF5 反演结果写入
│       ├── scripts/
│       │   └── run_inversion.py      # 主入口脚本
│       └── requirements.txt
└── DEPLOYMENT.md                     # 本文档
```

---

## 2. 依赖环境

### 2.1 编译器与核心库

| 依赖 | 最低版本 | 用途 |
|------|----------|------|
| GCC / Clang / MSVC | GCC 9+ / Clang 12+ / MSVC 2019+ | C++17 编译 |
| CMake | 3.18+ | 构建系统 |
| MPI (OpenMPI / MPICH / MS-MPI) | 2.0+ | 并行通信 |
| HDF5 (C++ 接口) | 1.10+ | 数据 I/O |
| Python | 3.8+ | 协调层运行时 |
| pybind11 | 2.10+ | C++/Python 绑定 |

### 2.2 Python 包

```bash
pip install numpy>=1.21 scipy>=1.7 h5py>=3.6 pyyaml>=6.0 mpi4py>=3.1
```

---

## 3. 编译指南

### 3.1 Linux (GCC + OpenMPI)

```bash
# 加载模块（超算环境）
module load gcc/11.2 openmpi/4.1 hdf5/1.12 cmake/3.22 python/3.9

# 克隆并进入项目
cd 03-seismic-tomography-compute

# 配置
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MPI=ON \
    -DENABLE_HDF5=ON \
    -DENABLE_PYBIND=ON \
    -DENABLE_AVX2=ON \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math"

# 编译
make -j$(nproc)

# 安装（可选）
make install
```

### 3.2 Linux (Intel oneAPI + Intel MPI)

```bash
module load intel/2022.1 intelmpi/2021.6 hdf5/1.12 cmake/3.22

mkdir build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MPI=ON \
    -DENABLE_HDF5=ON \
    -DENABLE_PYBIND=ON \
    -DENABLE_AVX2=ON \
    -DCMAKE_CXX_FLAGS="-O3 -xHost -ipo -ffast-math"

make -j$(nproc)
```

### 3.3 Windows (MSVC + MS-MPI)

```powershell
# 确保已安装 Visual Studio 2019+, MS-MPI, HDF5, Python

mkdir build
cd build

cmake .. `
    -G "Visual Studio 16 2019" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DENABLE_MPI=ON `
    -DENABLE_HDF5=ON `
    -DENABLE_PYBIND=ON `
    -DENABLE_AVX2=ON

cmake --build . --config Release --parallel
```

### 3.4 关键 CMake 选项说明

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_MPI` | ON | 启用 MPI 并行计算 |
| `ENABLE_HDF5` | ON | 启用 HDF5 数据 I/O |
| `ENABLE_PYBIND` | ON | 编译 pybind11 Python 绑定模块 |
| `ENABLE_AVX2` | ON | 启用 AVX2/FMA SIMD 指令集优化 |
| `ENABLE_TESTS` | OFF | 编译测试可执行文件 |

---

## 4. 超算集群部署

### 4.1 SLURM 作业提交

项目提供开箱即用的 SLURM 作业脚本 `deploy/slurm_job.sh`：

```bash
# 创建日志目录
mkdir -p logs

# 提交作业（4 节点 × 32 核 = 128 MPI 进程）
sbatch deploy/slurm_job.sh

# 查看作业状态
squeue -u $USER

# 查看输出
cat logs/seismic_<JOBID>.out
```

**关键 SLURM 参数调整：**

```bash
#SBATCH --nodes=4              # 计算节点数
#SBATCH --ntasks-per-node=32   # 每节点 MPI 进程数
#SBATCH --mem=64G              # 每节点内存
#SBATCH --time=24:00:00        # 最大运行时间
```

### 4.2 PBS/Torque 适配

```bash
# PBS 作业脚本示例
#PBS -N seismic_tomo
#PBS -l nodes=4:ppn=32
#PBS -l walltime=24:00:00
#PBS -q normal

cd $PBS_O_WORKDIR

mpirun -np 128 -machinefile $PBS_NODEFILE \
    python3 src/python/scripts/run_inversion.py \
    --nx 200 --ny 200 --nz 100 \
    --dx 500 --dy 500 --dz 200 \
    --max-iter 50 \
    --output-dir ./output/run_$PBS_JOBID
```

### 4.3 性能调优建议

#### MPI 进程映射

```bash
# 绑核运行（推荐）
mpirun -np 128 --bind-to core --map-by core ...

# 超线程环境下的槽位映射
mpirun -np 64 --bind-to hwthread --map-by slot ...
```

#### 内存对齐优化

系统内建以下内存对齐策略：

- **CSR 矩阵行对齐**：每行非零元素数按 64 字节（AVX2 向量寄存器宽度）对齐，确保 SpMV 内循环无跨缓存行访问
- **数据数组 64 字节对齐**：使用 `_mm_malloc` 替代标准 `malloc`，保证 `AlignedVector` 起始地址对齐
- **Cache-blocking**：稀疏矩阵自动按缓存行大小分块（`optimize_cache_blocking`），减少 TLB miss
- **软件预取**：`prefetch_row` 方法对即将访问的行发出 `_mm_prefetch(T0)` 指令

#### 大页内存（Huge Pages）

```bash
# Linux 启用 2MB 透明大页
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# 或使用 libhugetlbfs 显式大页
export HUGETLB_MORECORE=yes
```

---

## 5. 运行参数

### 5.1 命令行参数

```bash
python src/python/scripts/run_inversion.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nx/--ny/--nz` | 100/100/50 | 三维网格尺寸 |
| `--dx/--dy/--dz` | 500/500/200 | 网格间距（米） |
| `--dt` | 0.001 | 时间步长（秒） |
| `--max-iter` | 50 | 最大反演迭代次数 |
| `--step-length` | 0.01 | 初始步长 |
| `--smoothing-sigma` | 2.0 | 梯度高斯平滑 sigma |
| `--regularization` | 1e-4 | Tikhonov 正则化参数 |
| `--misfit-type` | cross_correlation | 目标函数类型 |
| `--water-depth` | 4000.0 | 海水深度（米） |
| `--n-sources` | 10 | 合成震源数量 |
| `--n-stations` | 20 | OBS 台站数量 |
| `--output-dir` | ./output | 输出目录 |

### 5.2 YAML 配置文件

编辑 `config/default.yaml` 可覆盖默认参数，C++ 引擎启动时自动加载。

---

## 6. 输出数据格式

所有结果写入 HDF5 文件，结构如下：

```
inversion_results.h5
├── velocity_model/
│   ├── iter_0000/          # 第 0 次迭代速度模型 (nx×ny×nz, float64, gzip 压缩)
│   ├── iter_0001/
│   └── ...
├── gradient/
│   ├── iter_0000/          # 第 0 次迭代梯度
│   └── ...
├── wavefield/
│   ├── step_000000/        # 波场快照
│   └── ...
├── bathymetry              # 海底地形数据 (nx×ny, float64)
├── source_positions        # 震源坐标 (n_src×3, float64)
├── receiver_positions      # 接收器坐标 (n_rcv×3, float64)
├── observed_travel_times   # 观测走时 (float64)
├── calculated_travel_times # 计算走时 (float64)
├── misfit_history          # 目标函数历史 (float64)
├── iteration_history       # 迭代号 (int32)
└── config/                 # 运行参数快照
```

---

## 7. 不规则海底地形边界条件

系统通过 `TerrainBoundary` 类严格处理深水海底的不规则地形：

1. **海底地形加载**：支持从 OBS 实测数据点通过双线性插值生成全局海底深度网格
2. **合成地形生成**：`generate_synthetic_bathymetry()` 基于梯度+粗糙度参数生成逼真海底地形
3. **速度掩膜更新**：`update_velocity_mask()` 根据地形自动区分水层（Vp=1500 m/s）和沉积层（深度梯度模型）
4. **自由表面边界**：海面处施加零应力条件
5. **海底边界**：海底界面处强制波场归零，模拟水-固界面波阻抗差异
6. **边界掩膜**：`build_boundary_mask()` 标记每个网格点的水/固状态，供 FD 模板动态切换

---

## 8. 稀疏矩阵内存对齐优化

针对超大尺度稀疏矩阵的极致优化策略：

| 优化技术 | 实现位置 | 效果 |
|----------|----------|------|
| 64 字节 SIMD 对齐分配 | `aligned_alloc.h` | 确保 AVX2 load/store 无越界 |
| CSR 行非零元素对齐填充 | `csr_matrix.cpp:pad_to_alignment()` | 消除 FD 模板内循环跨缓存行 |
| Cache-blocking 行分块 | `csr_matrix.cpp:optimize_cache_blocking()` | 减少 L2/L3 cache miss |
| 软件预取指令 | `csr_matrix.cpp:prefetch_row()` | 隐藏内存延迟 |
| SpMV 双路累加 | `csr_matrix.cpp:spmv()` | 减少循环依赖，利于流水线 |
| Jacobi 预条件 CG | `sparse_ops.cpp:jacobi_preconditioned_cg()` | 加速 Hessian 近似系统收敛 |

---

## 9. 伴随状态法反演流程

```
┌──────────────────────────────────────────────────┐
│  初始速度模型 m₀                                  │
│       ↓                                          │
│  ┌────────────────────────────────────────┐      │
│  │  正传播 (Forward Propagation)           │      │
│  │  C++ MPI: 3D FD 波动方程求解            │      │
│  │  → 存储正传波场快照                      │      │
│  └───────────────┬────────────────────────┘      │
│                  ↓                               │
│  ┌────────────────────────────────────────┐      │
│  │  走时残差计算                            │      │
│  │  δt = t_obs - t_calc                    │      │
│  └───────────────┬────────────────────────┘      │
│                  ↓                               │
│  ┌────────────────────────────────────────┐      │
│  │  伴随传播 (Adjoint Propagation)          │      │
│  │  C++ MPI: 以残差为源反传                 │      │
│  └───────────────┬────────────────────────┘      │
│                  ↓                               │
│  ┌────────────────────────────────────────┐      │
│  │  梯度计算                                │      │
│  │  ∂J/∂m = -∫ λ ∂²u/∂m² dt              │      │
│  │  → 高斯平滑 + Tikhonov 正则化            │      │
│  └───────────────┬────────────────────────┘      │
│                  ↓                               │
│  ┌────────────────────────────────────────┐      │
│  │  线搜索 + 模型更新                       │      │
│  │  m_{k+1} = m_k - α ∇J                   │      │
│  │  → 写入 HDF5 数据集                      │      │
│  └───────────────┬────────────────────────┘      │
│                  ↓                               │
│           收敛? → 否 → 回到正传播                  │
│                  ↓ 是                            │
│           最终速度模型 m*                         │
└──────────────────────────────────────────────────┘
```

---

## 10. 常见问题

### Q: 编译时找不到 MPI

```bash
export MPI_HOME=/path/to/mpi
cmake .. -DMPI_HOME=${MPI_HOME}
```

### Q: HDF5 C++ 接口找不到

```bash
cmake .. -DHDF5_ROOT=/path/to/hdf5 -DHDF5_REQUIRE_CXX=ON
```

### Q: pybind11 编译失败

```bash
pip install pybind11
cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

### Q: 运行时 `ImportError: seismic_engine`

确保编译产出的 `.so` / `.pyd` 文件位于 Python 搜索路径中：

```bash
export PYTHONPATH=/path/to/build:$PYTHONPATH
```

或在 CMake 中执行 `make install` 将模块安装到 `site-packages`。

### Q: 大规模运行 OOM

1. 减小每节点 MPI 进程数（增加每进程可用内存）
2. 启用大页内存
3. 减小网格规模或增大网格间距
4. 使用 `sponge_width` 替代全空间吸收边界
