import numpy as np
from typing import Optional, Dict, Callable
import logging
import time

from .obs_data import OBSDataCoordinator
from .gradient import GradientComputer
from .model import VelocityModel
from .hdf5_writer import InversionHDF5Writer

logger = logging.getLogger(__name__)


class AdjointStateInversion:
    def __init__(self, model: VelocityModel,
                 obs_data: OBSDataCoordinator,
                 hdf5_writer: InversionHDF5Writer,
                 dt: float = 0.001,
                 max_iterations: int = 50,
                 step_length: float = 0.01,
                 smoothing_sigma: float = 2.0,
                 regularization_lambda: float = 1e-4,
                 misfit_tolerance: float = 1e-6,
                 misfit_type: str = "cross_correlation",
                 line_search_max: int = 10,
                 line_search_factor: float = 0.5):
        self.model = model
        self.obs_data = obs_data
        self.hdf5_writer = hdf5_writer
        self.dt = dt
        self.max_iterations = max_iterations
        self.step_length = step_length
        self.misfit_tolerance = misfit_tolerance
        self.misfit_type = misfit_type
        self.line_search_max = line_search_max
        self.line_search_factor = line_search_factor

        self.gradient_computer = GradientComputer(
            nx=model.nx, ny=model.ny, nz=model.nz,
            smoothing_sigma=smoothing_sigma,
            regularization_lambda=regularization_lambda,
        )

        self._current_iteration = 0
        self._current_misfit = float('inf')
        self._misfit_history: list = []
        self._converged = False

    def run(self, callback: Optional[Callable] = None) -> Dict:
        logger.info("Starting adjoint-state seismic tomography inversion")
        logger.info(self.model.summary())
        logger.info(self.obs_data.summary())

        self.hdf5_writer.write_config({
            'nx': self.model.nx,
            'ny': self.model.ny,
            'nz': self.model.nz,
            'dx': self.model.dx,
            'dy': self.model.dy,
            'dz': self.model.dz,
            'dt': self.dt,
            'max_iterations': self.max_iterations,
            'step_length': self.step_length,
            'misfit_type': self.misfit_type,
        })

        src_pos = self.obs_data.get_source_positions()
        rcv_pos = self.obs_data.get_receiver_positions()
        if len(src_pos) > 0 and len(rcv_pos) > 0:
            self.hdf5_writer.write_source_receiver_config(src_pos, rcv_pos)

        self._current_misfit = float('inf')

        for iteration in range(self.max_iterations):
            self._current_iteration = iteration
            t_start = time.time()

            logger.info(f"--- Iteration {iteration + 1}/{self.max_iterations} ---")

            calculated_times = self._forward_model()
            misfit = self.obs_data.compute_misfit(calculated_times, self.misfit_type)
            residuals = self.obs_data.compute_weighted_residuals(calculated_times)

            logger.info(f"  Misfit: {misfit:.6e}")
            self._misfit_history.append(misfit)
            self.hdf5_writer.record_misfit(iteration, misfit)

            if iteration > 0:
                rel_change = abs(self._misfit_history[-2] - misfit) / max(abs(self._misfit_history[-2]), 1e-30)
                if rel_change < self.misfit_tolerance:
                    logger.info(f"  Converged: relative misfit change {rel_change:.2e} < {self.misfit_tolerance}")
                    self._converged = True
                    break

            gradient = self._compute_gradient(residuals)
            gradient = self.gradient_computer.apply_smoothing(gradient)
            gradient = self.gradient_computer.apply_regularization(
                self.model.vp, gradient
            )
            gradient = self.gradient_computer.normalize_gradient(gradient)

            optimal_step = self._line_search(gradient, misfit)

            vp_update = self.model.flatten_vp() - optimal_step * gradient
            self.model.update_from_flat(vp_update)

            self.hdf5_writer.write_velocity_model(
                self.model.vp, self.model.nx, self.model.ny, self.model.nz,
                self.model.dx, self.model.dy, self.model.dz, iteration
            )
            self.hdf5_writer.write_gradient(
                gradient, self.model.nx, self.model.ny, self.model.nz, iteration
            )

            elapsed = time.time() - t_start
            logger.info(f"  Step length: {optimal_step:.6f}, Time: {elapsed:.2f}s")

            if callback is not None:
                callback(iteration, misfit, gradient, optimal_step)

        self.hdf5_writer.write_inversion_history()

        result = {
            'converged': self._converged,
            'iterations': self._current_iteration + 1,
            'final_misfit': self._misfit_history[-1] if self._misfit_history else float('inf'),
            'misfit_history': self._misfit_history,
        }

        logger.info(f"Inversion complete: {result['iterations']} iterations, "
                     f"final misfit: {result['final_misfit']:.6e}")
        return result

    def _forward_model(self) -> np.ndarray:
        vp = self.model.flatten_vp()
        n_picks = len(self.obs_data.picks)
        calculated_times = np.zeros(n_picks)

        for i, pick in enumerate(self.obs_data.picks):
            src = self.obs_data.sources.get(pick.source_id)
            sta = self.obs_data.stations.get(pick.station_id)
            if src is None or sta is None:
                continue

            dist = np.sqrt((src.x - sta.x)**2 +
                           (src.y - sta.y)**2 +
                           (src.z - sta.z)**2)

            iz_src = max(0, min(int((src.z - self.model.origin_z) / self.model.dz), self.model.nz - 1))
            iy_src = max(0, min(int((src.y - self.model.origin_y) / self.model.dy), self.model.ny - 1))
            ix_src = max(0, min(int((src.x - self.model.origin_x) / self.model.dx), self.model.nx - 1))

            iz_sta = max(0, min(int((sta.z - self.model.origin_z) / self.model.dz), self.model.nz - 1))
            iy_sta = max(0, min(int((sta.y - self.model.origin_y) / self.model.dy), self.model.ny - 1))
            ix_sta = max(0, min(int((sta.x - self.model.origin_x) / self.model.dx), self.model.nx - 1))

            n_steps = max(int(dist / (self.model.dx * 5)), 2)
            travel_time = 0.0

            for step in range(n_steps):
                frac = (step + 0.5) / n_steps
                ix = int(ix_src + frac * (ix_sta - ix_src))
                iy = int(iy_src + frac * (iy_sta - iy_src))
                iz = int(iz_src + frac * (iz_sta - iz_src))

                ix = max(0, min(ix, self.model.nx - 1))
                iy = max(0, min(iy, self.model.ny - 1))
                iz = max(0, min(iz, self.model.nz - 1))

                v_local = self.model.vp[ix, iy, iz]
                ds = dist / n_steps
                travel_time += ds / max(v_local, 100.0)

            calculated_times[i] = travel_time

        return calculated_times

    def _compute_gradient(self, residuals: np.ndarray) -> np.ndarray:
        n = self.model.total_cells()
        gradient = np.zeros(n)

        for i, pick in enumerate(self.obs_data.picks):
            src = self.obs_data.sources.get(pick.source_id)
            sta = self.obs_data.stations.get(pick.station_id)
            if src is None or sta is None:
                continue

            iz_src = max(0, min(int((src.z - self.model.origin_z) / self.model.dz), self.model.nz - 1))
            iy_src = max(0, min(int((src.y - self.model.origin_y) / self.model.dy), self.model.ny - 1))
            ix_src = max(0, min(int((src.x - self.model.origin_x) / self.model.dx), self.model.nx - 1))

            iz_sta = max(0, min(int((sta.z - self.model.origin_z) / self.model.dz), self.model.nz - 1))
            iy_sta = max(0, min(int((sta.y - self.model.origin_y) / self.model.dy), self.model.ny - 1))
            ix_sta = max(0, min(int((sta.x - self.model.origin_x) / self.model.dx), self.model.nx - 1))

            dist = np.sqrt((src.x - sta.x)**2 + (src.y - sta.y)**2 + (src.z - sta.z)**2)
            n_steps = max(int(dist / (self.model.dx * 5)), 2)
            ds = dist / n_steps

            for step in range(n_steps):
                frac = (step + 0.5) / n_steps
                ix = int(ix_src + frac * (ix_sta - ix_src))
                iy = int(iy_src + frac * (iy_sta - iy_src))
                iz = int(iz_src + frac * (iz_sta - iz_src))

                ix = max(0, min(ix, self.model.nx - 1))
                iy = max(0, min(iy, self.model.ny - 1))
                iz = max(0, min(iz, self.model.nz - 1))

                idx = ix * self.model.ny * self.model.nz + iy * self.model.nz + iz
                v_local = self.model.vp[ix, iy, iz]
                frechet = -ds / (v_local * v_local)
                gradient[idx] += residuals[i] * frechet * pick.weight

        return gradient

    def _line_search(self, gradient: np.ndarray,
                      current_misfit: float) -> float:
        step = self.step_length

        for _ in range(self.line_search_max):
            trial_vp = self.model.flatten_vp() - step * gradient
            trial_model = self.model.vp.copy()
            self.model.update_from_flat(trial_vp)

            trial_times = self._forward_model()
            trial_misfit = self.obs_data.compute_misfit(trial_times, self.misfit_type)

            self.model.vp = trial_model

            if trial_misfit < current_misfit:
                return step

            step *= self.line_search_factor

        return step
