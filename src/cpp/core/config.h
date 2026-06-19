#pragma once

#include <string>
#include <map>
#include <vector>

namespace seismic {

struct SolverConfig {
    double frequency;
    double dt;
    int nt;
    double water_depth;
    double seafloor_gradient;
    int sponge_width;
    double sponge_damping;
    int snapshot_interval;
    std::string output_dir;
};

struct ModelConfig {
    double vp_min;
    double vp_max;
    double vs_min;
    double vs_max;
    double rho_min;
    double rho_max;
    double initial_vp;
    double initial_vs;
    double initial_rho;
};

struct InversionConfig {
    int max_iterations;
    double step_length;
    double gradient_smoothing_sigma;
    double regularization_lambda;
    double misfit_tolerance;
    std::string misfit_type;
    int gradient_precondition;
    int line_search_max;
    double line_search_factor;
};

struct TomographyConfig {
    SolverConfig solver;
    ModelConfig model;
    InversionConfig inversion;
    std::map<std::string, std::string> obs_stations;
    std::vector<double> source_positions;
};

TomographyConfig load_config(const std::string& yaml_path);

}
