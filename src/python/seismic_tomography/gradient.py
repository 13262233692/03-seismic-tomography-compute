import numpy as np
from typing import Optional

try:
    from seismic_engine import CSRMatrix, SparseOps
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False


class GradientComputer:
    def __init__(self, nx: int, ny: int, nz: int,
                 smoothing_sigma: float = 2.0,
                 regularization_lambda: float = 1e-4):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.smoothing_sigma = smoothing_sigma
        self.regularization_lambda = regularization_lambda

        self._gradient: Optional[np.ndarray] = None
        self._preconditioner: Optional[np.ndarray] = None

    def compute_adjoint_gradient(self,
                                  forward_wavefield: np.ndarray,
                                  adjoint_wavefield: np.ndarray,
                                  velocity: np.ndarray,
                                  dt: float) -> np.ndarray:
        n = self.nx * self.ny * self.nz
        gradient = np.zeros(n)

        fwd = forward_wavefield.reshape(-1)
        adj = adjoint_wavefield.reshape(-1)
        vel = velocity.reshape(-1)

        second_derivative = 2.0 * vel * dt * dt
        gradient = -fwd * adj * second_derivative

        self._gradient = gradient
        return gradient

    def compute_frechet_gradient(self,
                                  residual: np.ndarray,
                                  sensitivity_matrix) -> np.ndarray:
        if HAS_CPP_ENGINE and isinstance(sensitivity_matrix, CSRMatrix):
            x = np.ascontiguousarray(residual, dtype=np.float64)
            gradient = SparseOps.dot_product(
                sensitivity_matrix, x
            )
        else:
            gradient = sensitivity_matrix.T @ residual

        self._gradient = gradient
        return gradient

    def apply_smoothing(self, gradient: Optional[np.ndarray] = None) -> np.ndarray:
        if gradient is None:
            gradient = self._gradient
        if gradient is None:
            raise ValueError("No gradient computed yet")

        if HAS_CPP_ENGINE:
            g = np.ascontiguousarray(gradient, dtype=np.float64)
            smoothed = SparseOps.smooth_gradient(
                g, self.nx, self.ny, self.nz, self.smoothing_sigma
            )
            self._gradient = smoothed
            return smoothed
        else:
            return self._apply_gaussian_smoothing_numpy(gradient)

    def _apply_gaussian_smoothing_numpy(self, gradient: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        g3d = gradient.reshape(self.nx, self.ny, self.nz)
        smoothed = gaussian_filter(g3d, sigma=self.smoothing_sigma)
        self._gradient = smoothed.ravel()
        return self._gradient

    def apply_preconditioning(self, gradient: Optional[np.ndarray] = None,
                               preconditioner: Optional[np.ndarray] = None) -> np.ndarray:
        if gradient is None:
            gradient = self._gradient
        if gradient is None:
            raise ValueError("No gradient computed yet")

        if preconditioner is not None:
            self._preconditioner = preconditioner

        if self._preconditioner is not None:
            gradient = gradient * self._preconditioner

        self._gradient = gradient
        return gradient

    def build_depth_preconditioner(self, water_depth_iz: int,
                                    dz: float = 200.0) -> np.ndarray:
        n = self.nx * self.ny * self.nz
        precond = np.ones(n)

        for iz in range(self.nz):
            if iz < water_depth_iz:
                precond.reshape(self.nx, self.ny, self.nz)[:, :, iz] = 0.0
            else:
                depth_factor = 1.0 / (1.0 + (iz - water_depth_iz) * dz / 10000.0)
                precond.reshape(self.nx, self.ny, self.nz)[:, :, iz] = depth_factor

        self._preconditioner = precond
        return precond

    def apply_regularization(self, velocity: np.ndarray,
                              gradient: Optional[np.ndarray] = None) -> np.ndarray:
        if gradient is None:
            gradient = self._gradient
        if gradient is None:
            raise ValueError("No gradient computed yet")

        vel_3d = velocity.reshape(self.nx, self.ny, self.nz)
        laplacian = np.zeros_like(vel_3d)

        laplacian[1:-1, :, :] += vel_3d[2:, :, :] - 2*vel_3d[1:-1, :, :] + vel_3d[:-2, :, :]
        laplacian[:, 1:-1, :] += vel_3d[:, 2:, :] - 2*vel_3d[:, 1:-1, :] + vel_3d[:, :-2, :]
        laplacian[:, :, 1:-1] += vel_3d[:, :, 2:] - 2*vel_3d[:, :, 1:-1] + vel_3d[:, :, :-2]

        reg_gradient = gradient - self.regularization_lambda * laplacian.ravel()
        self._gradient = reg_gradient
        return reg_gradient

    def normalize_gradient(self, gradient: Optional[np.ndarray] = None) -> np.ndarray:
        if gradient is None:
            gradient = self._gradient
        if gradient is None:
            raise ValueError("No gradient computed yet")

        max_abs = np.max(np.abs(gradient))
        if max_abs > 1e-30:
            gradient = gradient / max_abs

        self._gradient = gradient
        return gradient

    @property
    def gradient(self) -> Optional[np.ndarray]:
        return self._gradient
