
#pragma once

#include "core/type.hpp"
#include "robot_env/environment.hpp"

#include <pybind11/eigen.h>
#include <vector>

#define USE_STEP_OMP
#define USE_RESET_OMP

namespace py = pybind11;


class RobotNavParallelEnv {
public:
    using State = typename RobotNavEnv::State;
    using Act = typename RobotNavEnv::Act;
    using Obs = typename RobotNavEnv::Obs;
    using Rew = typename RobotNavEnv::Rew;

private:
    RobotNavEnv _env;
    uint64_t _seed;
    int _N;
    bool _auto_reset;

public:
    RobotNavParallelEnv(uint64_t seed, int n, bool auto_reset) : _seed(seed), _N(n), _auto_reset(auto_reset) {}

    static constexpr int obs_dim() { return RobotNavEnv::obs_dim(); }
    static constexpr int act_dim() { return RobotNavEnv::act_dim(); }
    
    void reset(std::vector<State>& states, double* obs_out) {
        #ifdef USE_RESET_OMP
        #ifndef __INTELLISENSE__
        #pragma omp parallel for schedule(dynamic)
        #endif
        #endif
        for (int i = 0; i < _N; ++i) {
            _env.reset(states[i], _seed + i);
            _env.observe(states[i], obs_out + i * obs_dim());
        }
    }

    void step(std::vector<State>& states, const double* acts_in, double* obs_out, double* rew_out, double* term_out, double* trun_out, double* tau_out) {
        #ifdef USE_STEP_OMP
        #ifndef __INTELLISENSE__
        #pragma omp parallel for schedule(dynamic)
        #endif
        #endif
        for (int i = 0; i < _N; ++i) {
            auto r = _env.step(states[i], acts_in + i * act_dim());
            _env.observe(states[i], obs_out + i * obs_dim());
            rew_out[i] = r.rew; term_out[i] = (double)r.terminated; trun_out[i] = (double)r.truncated; tau_out[i] = r.tau;

            if (_auto_reset && (r.terminated || r.truncated)) _env.reset(states[i], states[i].rng);
        }
    }

    py::tuple render_state(const State& state) const {
        return _env.render_state(state);
    }

    py::dict config() const { return _env.config(); }
};