import argparse
import logging
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from seismic_tomography import (
    AdjointStateInversion,
    OBSDataCoordinator,
    VelocityModel,
    InversionHDF5Writer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep-sea seismic wave tomography inversion system"
    )
    parser.add_argument("--nx", type=int, default=100,
                        help="Grid size in X direction")
    parser.add_argument("--ny", type=int, default=100,
                        help="Grid size in Y direction")
    parser.add_argument("--nz", type=int, default=50,
                        help="Grid size in Z direction")
    parser.add_argument("--dx", type=float, default=500.0,
                        help="Grid spacing in X (meters)")
    parser.add_argument("--dy", type=float, default=500.0,
                        help="Grid spacing in Y (meters)")
    parser.add_argument("--dz", type=float, default=200.0,
                        help="Grid spacing in Z (meters)")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Time step (seconds)")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum inversion iterations")
    parser.add_argument("--step-length", type=float, default=0.01,
                        help="Initial step length for model update")
    parser.add_argument("--smoothing-sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma for gradient")
    parser.add_argument("--regularization", type=float, default=1e-4,
                        help="Tikhonov regularization parameter")
    parser.add_argument("--misfit-tolerance", type=float, default=1e-6,
                        help="Convergence tolerance for misfit change")
    parser.add_argument("--misfit-type", type=str, default="cross_correlation",
                        choices=["l2", "l1", "cross_correlation"],
                        help="Misfit function type")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for results")
    parser.add_argument("--water-depth", type=float, default=4000.0,
                        help="Water depth for initial model (meters)")
    parser.add_argument("--n-sources", type=int, default=10,
                        help="Number of synthetic sources")
    parser.add_argument("--n-stations", type=int, default=20,
                        help="Number of OBS stations")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(Path(args.output_dir) / "inversion.log"), mode='a'
            ),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Deep-Sea Seismic Wave Tomography Inversion System")
    logger.info("=" * 60)

    model = VelocityModel(
        nx=args.nx, ny=args.ny, nz=args.nz,
        dx=args.dx, dy=args.dy, dz=args.dz,
    )
    model.initialize_gradient_model(water_depth=args.water_depth)
    logger.info(model.summary())

    obs_data = OBSDataCoordinator()
    obs_data.generate_synthetic_data(
        n_sources=args.n_sources,
        n_stations=args.n_stations,
        domain_size=(args.nx * args.dx, args.ny * args.dy, args.nz * args.dz),
    )
    logger.info(obs_data.summary())

    hdf5_writer = InversionHDF5Writer(output_dir=args.output_dir)

    try:
        from seismic_engine import generate_bathymetry
        bathymetry = generate_bathymetry(
            args.nx, args.ny, args.dx, args.dy,
            0.0, 0.0, args.water_depth, 0.05, 100.0
        )
        model.apply_bathymetry(bathymetry)
        hdf5_writer.write_bathymetry(bathymetry, args.dx, args.dy)
        logger.info("Applied irregular bathymetry boundary from C++ engine")
    except ImportError:
        logger.warning("C++ engine not available, using flat bathymetry approximation")

    inversion = AdjointStateInversion(
        model=model,
        obs_data=obs_data,
        hdf5_writer=hdf5_writer,
        dt=args.dt,
        max_iterations=args.max_iter,
        step_length=args.step_length,
        smoothing_sigma=args.smoothing_sigma,
        regularization_lambda=args.regularization,
        misfit_tolerance=args.misfit_tolerance,
        misfit_type=args.misfit_type,
    )

    result = inversion.run()

    logger.info("=" * 60)
    logger.info("Inversion Results:")
    logger.info(f"  Converged: {result['converged']}")
    logger.info(f"  Iterations: {result['iterations']}")
    logger.info(f"  Final misfit: {result['final_misfit']:.6e}")
    logger.info(f"  Results saved to: {hdf5_writer.filepath_str}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
