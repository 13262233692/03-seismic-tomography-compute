import numpy as np
from typing import Optional, Dict, Callable
import logging
import time

from .obs_data import OBSDataCoordinator
from .gradient import GradientComputer
from .model import VelocityModel
from .hdf5_writer import InversionHDF5Writer

try:
    from seismic_engine import (
        check_field_health, clamp_field_safe, repair_field_nan,
        StepResult, HealthStatus,
    )
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False

logger = logging.getLogger(__name__)


class FaultRecoveryPolicy:
    def __init__(self, max_consecutive_faults: int = 5,
                 nan_ratio_threshold: float = 0.1,
                 step_reduction_factor: float = 0.5,
                 min_step_length: float = 1e-8,
                 velocity_clamp_range: tuple = (1450.0, 8500.0)):
        self.max_consecutive_faults = max_consecutive_faults
        self.nan_ratio_threshold = nan_ratio_threshold
        self.step_reduction_factor = step_reduction_factor
        self.min_step_length = min_step_length
        self.velocity_clamp_range = velocity_clamp_range
        self._consecutive_faults = 0
        self._total_faults = 0
        self._recovered_iterations = 0
        self._skipped_iterations = 0

    def should_continue(self, nan_ratio: float) -> bool:
        if self._consecutive_faults >= self.max_consecutive_faults:
            logger.error(
                f"Exceeded max consecutive faults ({self.max_consecutive_faults}). "
                f"Aborting inversion."
            )
            return False
        if nan_ratio > self.nan_ratio_threshold:
            logger.error(
                f"NaN ratio {nan_ratio:.4f} exceeds threshold "
                f"{self.nan_ratio_threshold}. Aborting."
            )
            return False
        return True

    def record_fault(self):
        self._consecutive_faults += 1
        self._total_faults += 1

    def record_recovery(self):
        self._recovered_iterations += 1
        self._consecutive_faults = 0

    def record_skip(self):
        self._skipped_iterations += 1

    def reduced_step_length(self, current_step: float) -> float:
        new_step = current_step * self.step_reduction_factor
        return max(new_step, self.min_step_length)

    def clamp_velocity(self, vp: np.ndarray) -> np.ndarray:
        lo, hi = self.velocity_clamp_range
        vp = np.where(np.isfinite(vp), vp, lo)
        return np.clip(vp, lo, hi)

    @property
    def stats(self) -> Dict:
        return {
            'total_faults': self._total_faults,
            'consecutive_faults': self._consecutive_faults,
            'recovered_iterations': self._recovered_iterations,
            'skipped_iterations': self._skipped_iterations,
        }


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
                 line_search_factor: float = 0.5,
                 enable_fault_tolerance: bool = True):
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
        self.enable_fault_tolerance = enable_fault_tolerance

        self.gradient_computer = GradientComputer(
            nx=model.nx, ny=model.ny, nz=model.nz,
            smoothing_sigma=smoothing_sigma,
            regularization_lambda=regularization_lambda,
        )

        self.fault_policy = FaultRecoveryPolicy()

        self._current_iteration = 0
        self._current_misfit = float('inf')
        self._misfit_history: list = []
        self._converged = False
        self._fault_events: list = []

    def run(self, callback: Optional[Callable] = None) -> Dict:
        logger.info("Starting adjoint-state seismic tomography inversion")
        logger.info(self.model.summary())
        logger.info(self.obs_data.summary())

        if self.enable_fault_tolerance:
            logger.info("Fault tolerance: ENABLED")
            self._verify_model_health()

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
            'fault_tolerance': self.enable_fault_tolerance,
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

            if self.enable_fault_tolerance:
                vp_flat = self.model.flatten_vp()
                nan_count = self._check_field_health(vp_flat)
                if nan_count > 0:
                    nan_ratio = nan_count / len(vp_flat)
                    logger.warning(
                        f"  Velocity model contains {nan_count} NaN/Inf cells "
                        f"(ratio={nan_ratio:.6f})"
                    )
                    if not self.fault_policy.should_continue(nan_ratio):
                        logger.error("Fault tolerance threshold exceeded. Stopping.")
                        break
                    self.fault_policy.record_fault()
                    vp_flat = self._repair_field(vp_flat)
                    self.model.update_from_flat(vp_flat)

            calculated_times = self._forward_model()
            misfit = self.obs_data.compute_misfit(calculated_times, self.misfit_type)

            if not np.isfinite(misfit):
                logger.warning(
                    f"  Non-finite misfit detected: {misfit}. "
                    f"Attempting fault recovery."
                )
                self.fault_policy.record_fault()
                self.step_length = self.fault_policy.reduced_step_length(self.step_length)
                vp_flat = self.model.flatten_vp()
                vp_flat = self.fault_policy.clamp_velocity(vp_flat)
                self.model.update_from_flat(vp_flat)
                self._record_fault_event(iteration, "non_finite_misfit", misfit)
                self.fault_policy.record_skip()
                continue

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

            if self.enable_fault_tolerance:
                grad_nan = np.sum(~np.isfinite(gradient))
                if grad_nan > 0:
                    logger.warning(
                        f"  Gradient contains {grad_nan} NaN/Inf values. Repairing."
                    )
                    gradient = np.where(np.isfinite(gradient), gradient, 0.0)
                    self.fault_policy.record_fault()
                    self._record_fault_event(iteration, "gradient_nan", int(grad_nan))

            gradient = self.gradient_computer.apply_smoothing(gradient)
            gradient = self.gradient_computer.apply_regularization(
                self.model.vp, gradient
            )
            gradient = self.gradient_computer.normalize_gradient(gradient)

            optimal_step = self._line_search(gradient, misfit)

            vp_update = self.model.flatten_vp() - optimal_step * gradient

            if self.enable_fault_tolerance:
                vp_update = self.fault_policy.clamp_velocity(vp_update)

            self.model.update_from_flat(vp_update)

            if self.enable_fault_tolerance:
                self.fault_policy.record_recovery()

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
            'fault_events': self._fault_events,
            'fault_stats': self.fault_policy.stats if self.enable_fault_tolerance else None,
        }

        if self.enable_fault_tolerance and self.fault_policy.stats:
            fs = self.fault_policy.stats
            logger.info(
                f"Fault tolerance summary: "
                f"{fs['total_faults']} faults, "
                f"{fs['recovered_iterations']} recovered, "
                f"{fs['skipped_iterations']} skipped"
            )

        logger.info(f"Inversion complete: {result['iterations']} iterations, "
                     f"final misfit: {result['final_misfit']:.6e}")
        return result

    def _verify_model_health(self):
        vp = self.model.flatten_vp()
        nan_count = np.sum(~np.isfinite(vp))
        neg_count = np.sum(vp < 0)

        if nan_count > 0 or neg_count > 0:
            logger.warning(
                f"Initial model health check: {nan_count} NaN/Inf, "
                f"{neg_count} negative velocities. Auto-repairing."
            )
            vp = np.where(np.isfinite(vp) & (vp > 0), vp, 1500.0)
            self.model.update_from_flat(vp)

        v_max = np.max(vp)
        cfl = v_max * self.dt / min(self.model.dx, self.model.dy, self.model.dz)
        if cfl > 0.5:
            logger.warning(
                f"CFL condition may be violated: CFL={cfl:.4f} > 0.5. "
                f"Consider reducing dt or increasing grid spacing."
            )

    def _check_field_health(self, data: np.ndarray) -> int:
        if HAS_CPP_ENGINE:
            try:
                return int(check_field_health(np.ascontiguousarray(data, dtype=np.float64)))
            except Exception:
                pass
        return int(np.sum(~np.isfinite(data)))

    def _repair_field(self, data: np.ndarray) -> np.ndarray:
        if HAS_CPP_ENGINE:
            try:
                repaired = np.ascontiguousarray(data.copy(), dtype=np.float64)
                repair_field_nan(
                    repaired,
                    self.model.nx, self.model.ny, self.model.nz,
                    halo_width=1, replace_value=1500.0
                )
                return repaired
            except Exception:
                pass

        data = np.where(np.isfinite(data), data, 1500.0)
        return data

    def _record_fault_event(self, iteration: int, fault_type: str, detail):
        event = {
            'iteration': iteration,
            'type': fault_type,
            'detail': str(detail),
            'timestamp': time.time(),
        }
        self._fault_events.append(event)
        logger.warning(f"  Fault recorded: iter={iteration}, type={fault_type}, detail={detail}")

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

                if not np.isfinite(v_local) or v_local <= 0:
                    v_local = 1500.0

                ds = dist / n_steps
                tt_contribution = ds / v_local

                if not np.isfinite(tt_contribution):
                    continue

                travel_time += tt_contribution

            if not np.isfinite(travel_time) or travel_time < 0:
                travel_time = dist / 6000.0

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

            if not np.isfinite(residuals[i]):
                continue

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

                if not np.isfinite(v_local) or v_local <= 0:
                    continue

                frechet = -ds / (v_local * v_local)

                if not np.isfinite(frechet):
                    continue

                gradient[idx] += residuals[i] * frechet * pick.weight

        return gradient

    def _line_search(self, gradient: np.ndarray,
                      current_misfit: float) -> float:
        step = self.step_length

        for _ in range(self.line_search_max):
            trial_vp = self.model.flatten_vp() - step * gradient

            if self.enable_fault_tolerance:
                trial_vp = self.fault_policy.clamp_velocity(trial_vp)

            trial_model = self.model.vp.copy()
            self.model.update_from_flat(trial_vp)

            trial_times = self._forward_model()
            trial_misfit = self.obs_data.compute_misfit(trial_times, self.misfit_type)

            self.model.vp = trial_model

            if not np.isfinite(trial_misfit):
                step *= self.fault_policy.step_reduction_factor
                continue

            if trial_misfit < current_misfit:
                return step

            step *= self.line_search_factor

        return step
