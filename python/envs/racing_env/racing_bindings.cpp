
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/runner.hpp"

namespace py = pybind11;


PYBIND11_MODULE(racing_env, m) {
    m.doc() = "Stateless C++ RL environment (CarRacingEnv)";
    
    // ---- CarRacingEnv ------------------------------------------------------
    py::class_<RacingEnv>(m, "RacingEnv")
        .def(py::init<std::uint64_t>())
        .def_static("num_agents", &RacingEnv::num_agents)
        .def_static("obs_dim", &RacingEnv::obs_dim)
        .def_static("act_dim", &RacingEnv::act_dim)
        .def("current_agents", &RacingEnv::current_agents)
        .def("reset", &RacingEnv::reset)
        .def("step", &RacingEnv::step, py::arg("action"))
        .def("render_state", &RacingEnv::render_state)
        .def("config", &RacingEnv::config);
    
    py::class_<VecRacingEnv>(m, "VecRacingEnv")
        .def(py::init<std::uint64_t, int>())
        .def_static("num_agents", &VecRacingEnv::num_agents)
        .def_static("obs_dim", &VecRacingEnv::obs_dim)
        .def_static("act_dim", &VecRacingEnv::act_dim)
        .def("current_agents", &VecRacingEnv::current_agents)
        .def("reset", &VecRacingEnv::reset)
        .def("step", &VecRacingEnv::step, py::arg("actions"))
        .def("render_state", &VecRacingEnv::render_state)
        .def("config", &VecRacingEnv::config);
}