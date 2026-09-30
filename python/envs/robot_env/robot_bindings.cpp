
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/runner.hpp"

namespace py = pybind11;


PYBIND11_MODULE(robot_env, m) {
    m.doc() = "Stateless C++ RL environment (RobotEnv)";

    // ---- RobotNavEnv ------------------------------------------------------
    py::class_<RobotEnv>(m, "RobotEnv")
        .def(py::init<std::uint64_t>())
        .def_static("obs_dim", &RobotEnv::obs_dim)
        .def_static("act_dim", &RobotEnv::act_dim)
        .def("reset", &RobotEnv::reset)
        .def("step", &RobotEnv::step, py::arg("action"))
        .def("render_state", &RobotEnv::render_state)
        .def("config", &RobotEnv::config);
    
    py::class_<VecRobotEnv>(m, "VecRobotEnv")
        .def(py::init<std::uint64_t, int>())
        .def_static("obs_dim", &VecRobotEnv::obs_dim)
        .def_static("act_dim", &VecRobotEnv::act_dim)
        .def("reset", &VecRobotEnv::reset)
        .def("step", &VecRobotEnv::step, py::arg("action"))
        .def("render_state", &VecRobotEnv::render_state)
        .def("config", &VecRobotEnv::config);
}