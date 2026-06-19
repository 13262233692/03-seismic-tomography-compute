import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class OBSPick:
    station_id: str
    source_id: str
    phase_type: str
    travel_time: float
    weight: float = 1.0
    residual: float = 0.0
    quality: int = 0


@dataclass
class OBSStation:
    station_id: str
    x: float
    y: float
    z: float
    picks: List[OBSPick] = field(default_factory=list)


@dataclass
class SeismicSource:
    source_id: str
    x: float
    y: float
    z: float
    origin_time: float = 0.0
    magnitude: float = 0.0


class OBSDataCoordinator:
    def __init__(self):
        self.stations: Dict[str, OBSStation] = {}
        self.sources: Dict[str, SeismicSource] = {}
        self.picks: List[OBSPick] = []
        self._src_rcv_pairs: List[Tuple[str, str]] = []

    def add_station(self, station_id: str, x: float, y: float, z: float):
        self.stations[station_id] = OBSStation(
            station_id=station_id, x=x, y=y, z=z
        )

    def add_source(self, source_id: str, x: float, y: float, z: float,
                   origin_time: float = 0.0, magnitude: float = 0.0):
        self.sources[source_id] = SeismicSource(
            source_id=source_id, x=x, y=y, z=z,
            origin_time=origin_time, magnitude=magnitude
        )

    def add_pick(self, station_id: str, source_id: str,
                 phase_type: str, travel_time: float,
                 weight: float = 1.0, quality: int = 0):
        pick = OBSPick(
            station_id=station_id,
            source_id=source_id,
            phase_type=phase_type,
            travel_time=travel_time,
            weight=weight,
            quality=quality,
        )
        self.picks.append(pick)
        if station_id in self.stations:
            self.stations[station_id].picks.append(pick)
        self._src_rcv_pairs.append((source_id, station_id))

    def load_from_sac(self, directory: str, station_list: List[str]):
        pass

    def load_from_seisan(self, filename: str):
        pass

    def load_picks_csv(self, filename: str):
        data = np.genfromtxt(filename, delimiter=',',
                             names=True, dtype=None, encoding='utf-8')
        for row in data:
            self.add_pick(
                station_id=str(row['station_id']),
                source_id=str(row['source_id']),
                phase_type=str(row['phase_type']),
                travel_time=float(row['travel_time']),
                weight=float(row.get('weight', 1.0)),
                quality=int(row.get('quality', 0)),
            )

    def get_picks_by_phase(self, phase_type: str) -> List[OBSPick]:
        return [p for p in self.picks if p.phase_type == phase_type]

    def get_picks_by_source(self, source_id: str) -> List[OBSPick]:
        return [p for p in self.picks if p.source_id == source_id]

    def get_picks_by_station(self, station_id: str) -> List[OBSPick]:
        return [p for p in self.picks if p.station_id == station_id]

    def compute_travel_time_residuals(self, calculated_times: np.ndarray) -> np.ndarray:
        if len(calculated_times) != len(self.picks):
            raise ValueError(
                f"Calculated times length {len(calculated_times)} "
                f"does not match picks count {len(self.picks)}"
            )
        residuals = np.zeros(len(self.picks))
        for i, pick in enumerate(self.picks):
            residuals[i] = pick.travel_time - calculated_times[i]
            self.picks[i].residual = residuals[i]
        return residuals

    def compute_weighted_residuals(self, calculated_times: np.ndarray) -> np.ndarray:
        residuals = self.compute_travel_time_residuals(calculated_times)
        weights = np.array([p.weight for p in self.picks])
        return residuals * weights

    def compute_misfit(self, calculated_times: np.ndarray,
                       misfit_type: str = "l2") -> float:
        residuals = self.compute_travel_time_residuals(calculated_times)
        weights = np.array([p.weight for p in self.picks])

        if misfit_type == "l2":
            return 0.5 * np.sum(weights * residuals ** 2)
        elif misfit_type == "l1":
            return np.sum(weights * np.abs(residuals))
        elif misfit_type == "cross_correlation":
            return 0.5 * np.sum(weights * residuals ** 2)
        else:
            raise ValueError(f"Unknown misfit type: {misfit_type}")

    def get_source_positions(self) -> np.ndarray:
        if not self.sources:
            return np.empty((0, 3))
        return np.array([[s.x, s.y, s.z] for s in self.sources.values()])

    def get_receiver_positions(self) -> np.ndarray:
        if not self.stations:
            return np.empty((0, 3))
        return np.array([[s.x, s.y, s.z] for s in self.stations.values()])

    def get_observed_travel_times(self, phase_type: str = "P") -> np.ndarray:
        picks = self.get_picks_by_phase(phase_type)
        return np.array([p.travel_time for p in picks])

    def get_pick_weights(self, phase_type: str = "P") -> np.ndarray:
        picks = self.get_picks_by_phase(phase_type)
        return np.array([p.weight for p in picks])

    def quality_filter(self, min_quality: int = 0, max_residual: float = float('inf')):
        filtered = []
        for pick in self.picks:
            if pick.quality >= min_quality and abs(pick.residual) <= max_residual:
                filtered.append(pick)
        return filtered

    def generate_synthetic_data(self, n_sources: int = 10, n_stations: int = 20,
                                domain_size: Tuple[float, float, float] = (50000, 50000, 20000),
                                noise_std: float = 0.05):
        np.random.seed(42)

        for i in range(n_stations):
            x = np.random.uniform(0, domain_size[0])
            y = np.random.uniform(0, domain_size[1])
            z = np.random.uniform(-domain_size[2], -500)
            self.add_station(f"OBS_{i:03d}", x, y, z)

        for i in range(n_sources):
            x = np.random.uniform(0, domain_size[0])
            y = np.random.uniform(0, domain_size[1])
            z = np.random.uniform(-domain_size[2], -5000)
            self.add_source(f"SRC_{i:03d}", x, y, z,
                            origin_time=0.0, magnitude=np.random.uniform(3.0, 6.0))

        avg_velocity = 6000.0
        for src in self.sources.values():
            for sta in self.stations.values():
                dist = np.sqrt((src.x - sta.x)**2 +
                               (src.y - sta.y)**2 +
                               (src.z - sta.z)**2)
                tt = dist / avg_velocity + np.random.normal(0, noise_std)
                if tt > 0:
                    self.add_pick(sta.station_id, src.source_id, "P", tt, weight=1.0)

    def summary(self) -> str:
        n_p_phases = len(self.get_picks_by_phase("P"))
        n_s_phases = len(self.get_picks_by_phase("S"))
        return (
            f"OBS Data Summary:\n"
            f"  Stations: {len(self.stations)}\n"
            f"  Sources: {len(self.sources)}\n"
            f"  Total picks: {len(self.picks)}\n"
            f"  P-phase picks: {n_p_phases}\n"
            f"  S-phase picks: {n_s_phases}\n"
            f"  Source-receiver pairs: {len(self._src_rcv_pairs)}"
        )
