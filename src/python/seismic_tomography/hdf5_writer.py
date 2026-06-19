import numpy as np
from typing import Optional, List, Dict
import h5py
from pathlib import Path

try:
    from seismic_engine import Hdf5IO as CppHdf5IO
    HAS_CPP_HDF5 = True
except ImportError:
    HAS_CPP_HDF5 = False


class InversionHDF5Writer:
    def __init__(self, output_dir: str, filename: str = "inversion_results.h5"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename
        self._misfits: List[float] = []
        self._iterations: List[int] = []
        self._step_lengths: List[float] = []

    def write_velocity_model(self, velocity: np.ndarray,
                              nx: int, ny: int, nz: int,
                              dx: float, dy: float, dz: float,
                              iteration: int):
        with h5py.File(str(self.filepath), 'a') as f:
            grp_name = f"velocity_model/iter_{iteration:04d}"
            if grp_name in f:
                del f[grp_name]

            ds = f.create_dataset(grp_name, data=velocity,
                                  compression="gzip", compression_opts=6)
            ds.attrs['nx'] = nx
            ds.attrs['ny'] = ny
            ds.attrs['nz'] = nz
            ds.attrs['dx'] = dx
            ds.attrs['dy'] = dy
            ds.attrs['dz'] = dz
            ds.attrs['iteration'] = iteration

    def write_gradient(self, gradient: np.ndarray,
                        nx: int, ny: int, nz: int,
                        iteration: int):
        with h5py.File(str(self.filepath), 'a') as f:
            grp_name = f"gradient/iter_{iteration:04d}"
            if grp_name in f:
                del f[grp_name]

            ds = f.create_dataset(grp_name, data=gradient,
                                  compression="gzip", compression_opts=4)
            ds.attrs['iteration'] = iteration

    def write_wavefield_snapshot(self, wavefield: np.ndarray,
                                  nx: int, ny: int, nz: int,
                                  timestep: int):
        with h5py.File(str(self.filepath), 'a') as f:
            ds_name = f"wavefield/step_{timestep:06d}"
            if ds_name in f:
                del f[ds_name]
            f.create_dataset(ds_name, data=wavefield,
                             compression="gzip", compression_opts=4)

    def write_bathymetry(self, bathymetry: np.ndarray,
                          dx: float, dy: float):
        with h5py.File(str(self.filepath), 'a') as f:
            if 'bathymetry' in f:
                del f['bathymetry']
            ds = f.create_dataset('bathymetry', data=bathymetry,
                                  compression="gzip")
            ds.attrs['dx'] = dx
            ds.attrs['dy'] = dy

    def write_source_receiver_config(self, source_positions: np.ndarray,
                                      receiver_positions: np.ndarray):
        with h5py.File(str(self.filepath), 'a') as f:
            if 'source_positions' in f:
                del f['source_positions']
            if 'receiver_positions' in f:
                del f['receiver_positions']
            f.create_dataset('source_positions', data=source_positions)
            f.create_dataset('receiver_positions', data=receiver_positions)

    def record_misfit(self, iteration: int, misfit: float, step_length: float = 0.0):
        self._misfits.append(misfit)
        self._iterations.append(iteration)
        self._step_lengths.append(step_length)

    def write_inversion_history(self):
        with h5py.File(str(self.filepath), 'a') as f:
            if 'misfit_history' in f:
                del f['misfit_history']
            if 'iteration_history' in f:
                del f['iteration_history']
            if 'step_length_history' in f:
                del f['step_length_history']

            f.create_dataset('misfit_history', data=np.array(self._misfits))
            f.create_dataset('iteration_history', data=np.array(self._iterations))
            f.create_dataset('step_length_history', data=np.array(self._step_lengths))

    def write_travel_times(self, observed: np.ndarray, calculated: np.ndarray,
                            source_ids: np.ndarray, receiver_ids: np.ndarray):
        with h5py.File(str(self.filepath), 'a') as f:
            for key in ['observed_travel_times', 'calculated_travel_times',
                        'source_ids', 'receiver_ids']:
                if key in f:
                    del f[key]

            f.create_dataset('observed_travel_times', data=observed)
            f.create_dataset('calculated_travel_times', data=calculated)
            f.create_dataset('source_ids', data=source_ids)
            f.create_dataset('receiver_ids', data=receiver_ids)

    def write_config(self, config: dict):
        with h5py.File(str(self.filepath), 'a') as f:
            if 'config' in f:
                del f['config']
            grp = f.create_group('config')
            for key, value in config.items():
                if isinstance(value, (int, float, str)):
                    grp.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    grp.create_dataset(key, data=np.array(value))

    def read_velocity_model(self, iteration: int) -> np.ndarray:
        with h5py.File(str(self.filepath), 'r') as f:
            ds_name = f"velocity_model/iter_{iteration:04d}"
            return f[ds_name][:]

    def read_inversion_history(self) -> Dict:
        with h5py.File(str(self.filepath), 'r') as f:
            return {
                'misfits': f['misfit_history'][:],
                'iterations': f['iteration_history'][:],
                'step_lengths': f['step_length_history'][:],
            }

    @property
    def filepath_str(self) -> str:
        return str(self.filepath)
