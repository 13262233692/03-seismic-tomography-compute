import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    from seismic_engine import compute_fd_stencil, CSRMatrix, SparseOps
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False


class VelocityModel:
    def __init__(self, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float,
                 origin_x: float = 0.0, origin_y: float = 0.0,
                 origin_z: float = 0.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z

        self.vp = np.ones((nx, ny, nz), dtype=np.float64) * 1500.0
        self.vs = np.zeros((nx, ny, nz), dtype=np.float64)
        self.rho = np.ones((nx, ny, nz), dtype=np.float64) * 2200.0

        self.vp_min = 1450.0
        self.vp_max = 8500.0
        self.vs_min = 0.0
        self.vs_max = 5000.0
        self.rho_min = 1000.0
        self.rho_max = 3300.0

        self._gradient = None
        self._fd_matrix: Optional[object] = None

    def initialize_constant(self, vp0: float, vs0: float = 0.0, rho0: float = 2200.0):
        self.vp[:] = vp0
        self.vs[:] = vs0
        self.rho[:] = rho0

    def initialize_gradient_model(self, water_vp: float = 1500.0,
                                   sediment_vp: float = 3000.0,
                                   mantle_vp: float = 8000.0,
                                   water_depth: float = 4000.0):
        for iz in range(self.nz):
            z = self.origin_z + iz * self.dz
            if z < water_depth:
                self.vp[:, :, iz] = water_vp
                self.vs[:, :, iz] = 0.0
                self.rho[:, :, iz] = 1025.0
            else:
                depth_below_sf = z - water_depth
                grad = 1.0 + 0.3 * depth_below_sf / 5000.0
                self.vp[:, :, iz] = min(sediment_vp * grad, self.vp_max)
                self.vs[:, :, iz] = min(self.vp[:, :, iz] / 1.732, self.vs_max)
                self.rho[:, :, iz] = min(2200.0 + 0.2 * depth_below_sf, self.rho_max)

    def apply_bathymetry(self, bathymetry: np.ndarray,
                          water_vp: float = 1500.0):
        if bathymetry.shape != (self.nx, self.ny):
            raise ValueError(
                f"Bathymetry shape {bathymetry.shape} does not match "
                f"model grid ({self.nx}, {self.ny})"
            )
        for ix in range(self.nx):
            for iy in range(self.ny):
                sf_iz = int(bathymetry[ix, iy] / self.dz)
                sf_iz = max(0, min(sf_iz, self.nz))
                self.vp[ix, iy, :sf_iz] = water_vp
                self.vs[ix, iy, :sf_iz] = 0.0
                self.rho[ix, iy, :sf_iz] = 1025.0

    def clamp_values(self):
        self.vp = np.where(np.isfinite(self.vp), self.vp, self.vp_min)
        self.vs = np.where(np.isfinite(self.vs), self.vs, self.vs_min)
        self.rho = np.where(np.isfinite(self.rho), self.rho, self.rho_min)
        self.vp = np.clip(self.vp, self.vp_min, self.vp_max)
        self.vs = np.clip(self.vs, self.vs_min, self.vs_max)
        self.rho = np.clip(self.rho, self.rho_min, self.rho_max)

    def total_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def flatten_vp(self) -> np.ndarray:
        return self.vp.ravel()

    def update_from_flat(self, vp_flat: np.ndarray):
        self.vp = vp_flat.reshape(self.nx, self.ny, self.nz)
        self.clamp_values()

    def build_fd_matrix(self, dt: float) -> object:
        if HAS_CPP_ENGINE:
            vp_flat = self.flatten_vp()
            self._fd_matrix = compute_fd_stencil(
                self.nx, self.ny, self.nz,
                vp_flat, self.dx, self.dy, self.dz, dt
            )
            return self._fd_matrix
        else:
            return self._build_fd_matrix_numpy(dt)

    def _build_fd_matrix_numpy(self, dt: float):
        from scipy import sparse
        n = self.total_cells()
        dx2 = 1.0 / (self.dx ** 2)
        dy2 = 1.0 / (self.dy ** 2)
        dz2 = 1.0 / (self.dz ** 2)
        dt2 = dt ** 2

        diags = []
        offsets = []

        main_diag = np.zeros(n)
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    idx = ix * self.ny * self.nz + iy * self.nz + iz
                    v2dt2 = self.vp[ix, iy, iz] ** 2 * dt2
                    main_diag[idx] = -2.0 * v2dt2 * (dx2 + dy2 + dz2) + 2.0
        diags.append(main_diag)
        offsets.append(0)

        def make_shift_diag(offset, coeff, boundary_check):
            d = np.zeros(n)
            for ix in range(self.nx):
                for iy in range(self.ny):
                    for iz in range(self.nz):
                        idx = ix * self.ny * self.nz + iy * self.nz + iz
                        if boundary_check(ix, iy, iz):
                            v2dt2 = self.vp[ix, iy, iz] ** 2 * dt2
                            d[idx] = v2dt2 * coeff
            return d

        diags.append(make_shift_diag(
            -1, dz2,
            lambda ix, iy, iz: iz > 0
        ))
        offsets.append(-1)

        diags.append(make_shift_diag(
            1, dz2,
            lambda ix, iy, iz: iz < self.nz - 1
        ))
        offsets.append(1)

        diags.append(make_shift_diag(
            -self.nz, dy2,
            lambda ix, iy, iz: iy > 0
        ))
        offsets.append(-self.nz)

        diags.append(make_shift_diag(
            self.nz, dy2,
            lambda ix, iy, iz: iy < self.ny - 1
        ))
        offsets.append(self.nz)

        diags.append(make_shift_diag(
            -self.ny * self.nz, dx2,
            lambda ix, iy, iz: ix > 0
        ))
        offsets.append(-self.ny * self.nz)

        diags.append(make_shift_diag(
            self.ny * self.nz, dx2,
            lambda ix, iy, iz: ix < self.nx - 1
        ))
        offsets.append(self.ny * self.nz)

        self._fd_matrix = sparse.diags(diags, offsets, shape=(n, n), format='csr')
        return self._fd_matrix

    def compute_gradient_norm(self) -> float:
        if self._gradient is None:
            return 0.0
        return float(np.linalg.norm(self._gradient))

    def summary(self) -> str:
        return (
            f"VelocityModel:\n"
            f"  Grid: {self.nx} x {self.ny} x {self.nz}\n"
            f"  Spacing: dx={self.dx}, dy={self.dy}, dz={self.dz}\n"
            f"  Vp range: [{self.vp.min():.1f}, {self.vp.max():.1f}] m/s\n"
            f"  Vs range: [{self.vs.min():.1f}, {self.vs.max():.1f}] m/s\n"
            f"  Rho range: [{self.rho.min():.1f}, {self.rho.max():.1f}] kg/m³"
        )
