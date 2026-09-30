
#pragma once

#include "car_env/parallel_env.hpp"
#include "robot_env/parallel_env.hpp"

constexpr const int DEFAULT_ENV_IDX = 0;

template <class Env>
class Runner {
    Env _env;
    std::vector<typename Env::State> _states;
    int _N;

public:
    Runner(uint64_t seed, int n_envs, bool auto_reset = true) : _env(seed, n_envs, auto_reset), _states(n_envs), _N(n_envs) {}

    py::array_t<double> reset() {
        py::array_t<double> obs({_N, obs_dim()});
        auto p = obs.mutable_data();
        {
            py::gil_scoped_release nogil;
            _env.reset(_states, p);
        }
        return obs;
    }

    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>> step(py::array_t<double> acts) {
        py::array_t<double> obs({_N, _env.obs_dim()});
        py::array_t<double> rew(_N), term(_N), trun(_N), tau(_N);

        double *obs_p = obs.mutable_data(); double *rew_p = rew.mutable_data();
        double *term_p = term.mutable_data(); double *trun_p = trun.mutable_data(); double *tau_p = tau.mutable_data();

        const double* act_p = acts.data();
        {
            py::gil_scoped_release nogil;
            _env.step(_states, act_p, obs_p, rew_p, term_p, trun_p, tau_p);
        }
        return {obs, rew, term, trun, tau};
    }

    py::array_t<double> current_agents() const {
        py::array_t<double> out(_N);
        double *p = out.mutable_data();

        _env.current_agents(_states, p);

        return out;
    }

    py::tuple render_state() const { return _env.render_state(_states[DEFAULT_ENV_IDX]); }
    py::dict config() const { return _env.config(); }
    
    static constexpr int obs_dim() { return Env::obs_dim(); }
    static constexpr int act_dim() { return Env::act_dim(); }
    static constexpr int num_agents() { return Env::num_agents(); }
};

using VecRacingEnv = Runner<CarRacingParallelEnv>;
struct RacingEnv : VecRacingEnv { RacingEnv(uint64_t s) : VecRacingEnv(s, 1) {} };

using VecRobotEnv = Runner<RobotNavParallelEnv>;
struct RobotEnv : VecRobotEnv { RobotEnv(uint64_t s) : VecRobotEnv(s, 1, false) {} };
