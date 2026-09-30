
#pragma once

#include "car_env/environment.hpp"

#include <pybind11/eigen.h>
#include <vector>

#define USE_OMP

namespace py = pybind11;


class CarRacingParallelEnv {
public:
    using State = typename CarRacingEnv::State;

private:
    CarRacingEnv _env;
    uint64_t _seed;
    int _N;
    bool _auto_reset;

public:
    CarRacingParallelEnv(uint64_t seed, int n, bool auto_reset = true) : _seed(seed), _N(n), _auto_reset(auto_reset) {}

    static constexpr int num_agents() { return CarRacingEnv::num_agents(); }
    static constexpr int obs_dim() { return CarRacingEnv::obs_dim(); }
    static constexpr int act_dim() { return CarRacingEnv::act_dim(); }

    void current_agents(const std::vector<State>& states, double* N_agents_out) const {
        for (int i = 0; i < _N; ++i) {
            N_agents_out[i] = states[i].cur_agent;
        }
    }

    void reset(std::vector<State>& states, double* obs_out) {
        #ifdef USE_OMP
        #ifndef __INTELLISENSE__
        #pragma omp parallel for schedule(dynamic)
        #endif
        #endif
        for (int i = 0; i < _N; ++i) {
            State& s = states[i];
            _env.reset(s, _seed + i);
            _env.observe(s, obs_out + i * obs_dim(), s.cur_agent);
        }
    }

    void step(std::vector<State>& states, const double* acts_in, double* obs_out, double* rew_out, double* term_out, double* trun_out, double* tau_out) {
        #ifdef USE_OMP
        #ifndef __INTELLISENSE__
        #pragma omp parallel for schedule(dynamic)
        #endif
        #endif
        for (int i = 0; i < _N; ++i) {
            State& s = states[i];

            if (_auto_reset && s.cur_agent < 0 && s.pending_terminal.empty()) {
                _env.reset(states[i], _seed + states[i].rng);
                _env.observe(s, obs_out + i * obs_dim(), s.cur_agent);
                rew_out[i] = 0.0; term_out[i] = 0.0; trun_out[i] = 0.0; tau_out[i] = 0.0;
                continue;
            }

            auto r = _env.step(s, acts_in + i * _env.act_dim());
            _env.observe(s, obs_out + i * obs_dim(), s.cur_agent);
            rew_out[i] = r.rew; term_out[i] = r.terminated; trun_out[i] = r.truncated; tau_out[i] = r.tau;
        }
    }

    py::tuple render_state(const State& state) const {
        return _env.render_state(state);
    }

    py::dict config() const { return _env.config(); }
};