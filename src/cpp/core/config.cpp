#include "config.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>

namespace seismic {

TomographyConfig load_config(const std::string& yaml_path) {
    TomographyConfig cfg;

    cfg.solver.frequency = 2.0;
    cfg.solver.dt = 0.001;
    cfg.solver.nt = 5000;
    cfg.solver.water_depth = 4000.0;
    cfg.solver.seafloor_gradient = 0.05;
    cfg.solver.sponge_width = 20;
    cfg.solver.sponge_damping = 0.015;
    cfg.solver.snapshot_interval = 100;
    cfg.solver.output_dir = "./output";

    cfg.model.vp_min = 1450.0;
    cfg.model.vp_max = 8500.0;
    cfg.model.vs_min = 0.0;
    cfg.model.vs_max = 5000.0;
    cfg.model.rho_min = 1000.0;
    cfg.model.rho_max = 3300.0;
    cfg.model.initial_vp = 1500.0;
    cfg.model.initial_vs = 0.0;
    cfg.model.initial_rho = 2200.0;

    cfg.inversion.max_iterations = 50;
    cfg.inversion.step_length = 0.01;
    cfg.inversion.gradient_smoothing_sigma = 2.0;
    cfg.inversion.regularization_lambda = 1e-4;
    cfg.inversion.misfit_tolerance = 1e-6;
    cfg.inversion.misfit_type = "cross_correlation";
    cfg.inversion.gradient_precondition = 1;
    cfg.inversion.line_search_max = 10;
    cfg.inversion.line_search_factor = 0.5;

    std::ifstream ifs(yaml_path);
    if (!ifs.is_open()) {
        return cfg;
    }

    std::string line;
    std::string section;
    while (std::getline(ifs, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0] == '#') continue;

        if (line.back() == ':') {
            section = line.substr(0, line.size() - 1);
            continue;
        }

        auto eq_pos = line.find(':');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string val = line.substr(eq_pos + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);

        if (section == "solver") {
            if (key == "frequency") cfg.solver.frequency = std::stod(val);
            else if (key == "dt") cfg.solver.dt = std::stod(val);
            else if (key == "nt") cfg.solver.nt = std::stoi(val);
            else if (key == "water_depth") cfg.solver.water_depth = std::stod(val);
            else if (key == "sponge_width") cfg.solver.sponge_width = std::stoi(val);
            else if (key == "sponge_damping") cfg.solver.sponge_damping = std::stod(val);
        } else if (section == "model") {
            if (key == "vp_min") cfg.model.vp_min = std::stod(val);
            else if (key == "vp_max") cfg.model.vp_max = std::stod(val);
            else if (key == "initial_vp") cfg.model.initial_vp = std::stod(val);
            else if (key == "initial_vs") cfg.model.initial_vs = std::stod(val);
            else if (key == "initial_rho") cfg.model.initial_rho = std::stod(val);
        } else if (section == "inversion") {
            if (key == "max_iterations") cfg.inversion.max_iterations = std::stoi(val);
            else if (key == "step_length") cfg.inversion.step_length = std::stod(val);
            else if (key == "gradient_smoothing_sigma")
                cfg.inversion.gradient_smoothing_sigma = std::stod(val);
            else if (key == "regularization_lambda")
                cfg.inversion.regularization_lambda = std::stod(val);
            else if (key == "misfit_type") cfg.inversion.misfit_type = val;
        }
    }

    return cfg;
}

}
