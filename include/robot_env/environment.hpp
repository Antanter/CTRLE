// ============================================================================
// environment.hpp
// Header-only *stateless* discrete-event (sMDP) environment, C++20 concepts.
// ============================================================================
#pragma once
#include "core/clock.hpp"
#include "core/event.hpp"
#include "core/environment.hpp"
#include <pybind11/eigen.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using Vec2 = Eigen::Vector2d;


namespace robot_cfg {
    constexpr const double DISTURBANCE_FORCE = 0.05;

    constexpr const double OBSTACLE_RADIUS = 0.25;
    constexpr const double GOAL_RADIUS = 0.3;

    constexpr double TIME_LIMIT = 30.0;
}


enum class EventKind : int { Decision, ControlExpiry, Disturbance, Timeout };

struct RobotEvent {
    double time{0.0};
    size_t seq{0};
    size_t ver{0};
    EventKind kind{EventKind::Decision};
};

static_assert(Event<RobotEvent>, "RobotEvent must satisfy Event");

using RobotEventQueue = EventQueue<RobotEvent>;


struct RobotState {
    Vec2 pos{0.0, 0.0};
    Vec2 vel{0.0, 0.0};
    Vec2 thrust{0.0, 0.0};
    Clock clock{};

    double t_last_decision{0.0};
    size_t seq{0};
    size_t dec_ver{0};
    size_t rng{0x9E3779B97F4A7C15ull};
    RobotEventQueue q{};

    Vec2 goal{0.0, 0.0};
    std::vector<Vec2> obstacles{};
    size_t n_dec{0};

    void push(double time, EventKind kind, size_t ver = 0) {
        q.push(RobotEvent{time, seq++, ver, kind});
    }
    double u01() {
        size_t z = (rng += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z =  z ^ (z >> 31);
        return (z >> 11) * (1.0 / 9007199254740992.0);
    }
    double expo(double mean) {
        return -mean * std::log(1.0 - u01());
    }
    double normal(double sd) {
        return sd * std::sqrt(-2.0 * std::log(std::max(u01(), 1e-12))) * std::cos(6.283185307179586 * u01());
    }
};

using RobotObs = std::vector<double>;
using RobotAct = std::vector<double>;
using RobotRew = double;


class RobotNavEnv {
public:
    using State = RobotState;
    using Act = RobotAct;
    using Obs = RobotObs;
    using Rew = RobotRew;

    static constexpr int obs_dim() { return 9 + 2 * K_OBS; }
    static constexpr int act_dim() { return 3; }

    RobotNavEnv(double obstacle_radius = robot_cfg::OBSTACLE_RADIUS, double goal_radius = robot_cfg::GOAL_RADIUS, double time_budget = robot_cfg::TIME_LIMIT)
        : _obstacle_r(obstacle_radius), _goal_r(goal_radius), _T(time_budget) {}
    
    void reset(State& s, uint64_t seed = 0) const {
        s = State{};

        if (seed) s.rng = seed | 1ull;

        s.pos = Vec2{ 0.5 + s.u01() * 4.0, 0.5 + s.u01() * 4.0 };
        s.goal = Vec2{4.5, 4.5};

        while ((s.pos - s.goal).norm() < 5.0) s.pos = Vec2{ 0.5 + s.u01() * 4.0, 0.5 + s.u01() * 4.0 };

        s.push(s.expo(_disturb_mean) + 1.0, EventKind::Disturbance);
        s.push(_T, EventKind::Timeout);

        s.obstacles.clear();
        for (int k = 0, tries = 0; k < _n_obs && tries < 20000; ++tries) {
            Vec2 c{ s.u01() * 5.0, s.u01() * 5.0 };
            if ((c - s.goal).norm() < 1.0 || (c - s.pos).norm() < 1.0) continue;

            bool ok = true;
            for (auto& o : s.obstacles) if ((c - o).norm() < 2*_obstacle_r + 0.2) { ok=false; break; }
            if (ok) { s.obstacles.push_back(c); ++k; }
        }
    }

    StepResult<Rew> step(State& s, const double* act_in) const {
        double reward = 0.0;

        s.thrust.x() = std::clamp(act_in[0], -1.0, 1.0);
        s.thrust.y() = std::clamp(act_in[1], -1.0, 1.0);
        const double commit = act_in[2];

        const std::uint64_t ver = ++s.dec_ver;
        ++s.n_dec;
        reward -= _dec_cost;

        s.push(s.clock.get() + commit_to_horizon(commit), EventKind::ControlExpiry, ver);

        while (!s.q.empty()) {
            RobotEvent e = s.q.top();
            s.q.pop();

            int hit = 0;
            reward += integrate(s, e.time, hit);

            if (hit != 0) {
                if (hit > 0) reward += 1.0; else reward -= 1.0;
                return finish(s, reward, true, false);
            }

            switch (e.kind) {
                case EventKind::Timeout:
                    return finish(s, reward, false, true);

                case EventKind::Disturbance:
                    s.vel.x() += s.normal(robot_cfg::DISTURBANCE_FORCE);
                    s.vel.y() += s.normal(robot_cfg::DISTURBANCE_FORCE);
                    s.push(s.clock.get() + s.expo(_disturb_mean), EventKind::Disturbance);
                    s.push(s.clock.get(), EventKind::Decision, s.dec_ver);
                    break;

                case EventKind::ControlExpiry:
                case EventKind::Decision:
                    if (e.ver != s.dec_ver) break;
                    return finish(s, reward, false, false);
            }
        }

        return finish(s, reward, false, true);
    }

    void observe(State& s, double* out) const {
        out[0]=s.pos.x(); out[1]=s.pos.y(); out[2]=s.vel.x(); out[3]=s.vel.y();
        out[4]=s.goal.x(); out[5]=s.goal.y(); out[6]=s.thrust.x(); out[7]=s.thrust.y();
        out[8]=_T - s.clock.get();

        auto obs = s.obstacles;
        std::sort(obs.begin(), obs.end(), [&](auto& a, auto& b){
            return (a - s.pos).squaredNorm() < (b - s.pos).squaredNorm();
        });

        for (int k = 0; k < K_OBS; ++k) {
            const int idx = 9+2*k;
            if (k < (int)obs.size()) {
                out[idx]=obs[k].x()-s.pos.x();
                out[idx+1]=obs[k].y()-s.pos.y();
            } else {
                out[idx]=0.0;
                out[idx+1]=0.0;
            }
        }
    }

    py::tuple render_state(const RobotState& s) const {
        py::array_t<double> robot(4);
        auto rb = robot.mutable_unchecked<1>();
        rb(0)=s.pos.x(); rb(1)=s.pos.y(); rb(2)=s.vel.x(); rb(3)=s.vel.y();

        py::array_t<double> goal(2);
        auto g = goal.mutable_unchecked<1>();
        g(0)=s.goal.x(); g(1)=s.goal.y();

        py::ssize_t M = s.obstacles.size();
        py::array_t<double> obst({M, (py::ssize_t)2});
        auto o = obst.mutable_unchecked<2>();
        for (py::ssize_t k=0;k<M;++k){ o(k,0)=s.obstacles[k].x(); o(k,1)=s.obstacles[k].y(); }

        return py::make_tuple(robot, goal, obst);
    }

    py::dict config() const {
        py::dict d;
        d["obstacle_r"]  = _obstacle_r;
        d["goal_r"]      = _goal_r;
        d["time_limit"]  = _T;
        d["world_min"]   = _wall_min;
        d["world_max"]   = _wall_max;
        d["wall_margin"] = _wall_margin;
        d["n_obstacles"] = _n_obs;
        d["drag"]        = _drag;
        d["h_min"]       = _h_min;
        d["h_max"]       = _h_max;
        return d;
    }

private:
    double commit_to_horizon(double commit) const {
        commit = std::tanh(commit);
        return _h_min + (commit + 1.0) * 0.5 * (_h_max - _h_min);
    }

    double integrate(State& s, double t_target, int& hit) const {
        constexpr double H = 0.02;
        double R = 0.0;

        while (s.clock.get() < t_target - 1e-12) {
            const double h = std::min(H, t_target - s.clock.get());
            s.vel += (_thrust_gain * s.thrust - _drag * s.vel) * h;
            s.pos += s.vel * h;
            if (s.pos.x() < _wall_min || s.pos.x() > _wall_max || s.pos.y() < _wall_min || s.pos.y() > _wall_max) { hit = -1; return R; }

            s.clock.add(h);
            const double dg = (s.pos - s.goal).norm();
            R += -0.005 * h;
            double d_wall = std::min({s.pos.x() - _wall_min, _wall_max - s.pos.x(), s.pos.y() - _wall_min, _wall_max - s.pos.y()});
            if (d_wall < _wall_margin) R += -_wall_pen * (_wall_margin - d_wall) / _wall_margin * h;

            const double r2 = _obstacle_r * _obstacle_r;
            for (auto& o : s.obstacles) if ((s.pos - o).squaredNorm() < r2) { hit = -1; return R; }
            if (dg < _goal_r) { hit = +1; return R; }
        }
        
        return R;
    }

    StepResult<Rew> finish(State& s, double reward, bool terminated, bool truncated) const {
        const double tau = s.clock.get() - s.t_last_decision;
        s.t_last_decision = s.clock.get();

        return StepResult<Rew>{reward, terminated, truncated, tau};
    }

    double _obstacle_r, _goal_r, _T;
    double _drag = 1.0;
    double _thrust_gain = 2.0;
    double _h_min = 0.2;
    double _h_max = 2.0;
    double _disturb_mean = 1.5;

    int _n_obs = 14;
    static constexpr int K_OBS = 14;
    double _dec_cost = 0.01;

    double _wall_min = 0.0;
    double _wall_max = 5.0;
    double _wall_margin = 0.3;
    double _wall_pen = 0.5;
};

static_assert(Environment<RobotNavEnv>);
