// ============================================================================
// racing_env.hpp
// Header-only stateless discrete-event (sMDP) MULTI-AGENT racing environment.
// ============================================================================
#pragma once
#include "core/clock.hpp"
#include "core/event.hpp"
#include "core/environment.hpp"
#include <pybind11/eigen.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using Vec2 = Eigen::Vector2d;

namespace racing_cfg {
    static constexpr int N_AGENTS = 2;
    static constexpr int K_ROCK = 12;

    static constexpr double TRACK_LEN = 60.0;
    static constexpr double TRACK_WIDTH = 4.0;
    static constexpr double ROCK_MEAN = 1.2;

    constexpr double TIME_LIMIT = 45.0;
}


enum class CarRaceEventKind : int { Decision, ControlExpiry, Spawn, Timeout };

struct RaceEvent {
    double time{0.0};
    size_t seq{0};
    size_t ver{0};
    int agent{-1};
    CarRaceEventKind kind{CarRaceEventKind::Decision};
};

using RaceEventQueue = EventQueue<RaceEvent>;


struct Car {
    double s{0.0};             // progress along track
    double d{0.0};             // lateral offset from centerline
    double vs{0.0};            // longitudinal velocity
    double vd{0.0};            // lateral velocity
    double thrust_s{0.0};      // commanded longitudinal accel (gas/brake)
    double thrust_d{0.0};      // commanded lateral accel (steer)
    size_t dec_ver{0};         // own decision version
    size_t n_dec{0};
    double rew_acc{0.0};
    bool alive{true};
    bool closed{false};
};

struct Rock {
    double s{0.0};
    double d{0.0};
};

struct RaceState {
    std::array<Car, racing_cfg::N_AGENTS> cars{};
    std::vector<Rock> rocks{};
    Clock clock{};

    size_t seq{0};
    size_t rng{0x9E3779B97F4A7C15ull};
    RaceEventQueue q{};
    std::vector<int> pending_terminal{};
    std::array<double, racing_cfg::N_AGENTS> t_last{};

    int cur_agent{-1};
    double s_spawn_frontier{0.0};

    void push(double time, CarRaceEventKind kind, int agent = -1, size_t ver = 0) {
        q.push(RaceEvent{time, seq++, ver, agent, kind});
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


using RaceObs = std::vector<double>;
using RaceAct = std::vector<double>;
using RaceRew = double;

class CarRacingEnv {
public:
    using State = RaceState;
    using Act = RaceAct;
    using Obs = RaceObs;
    using Rew = RaceRew;

    static constexpr int num_agents() { return racing_cfg::N_AGENTS; }
    static constexpr int obs_dim() { return 7 + 2 * racing_cfg::K_ROCK; }
    static constexpr int act_dim() { return 3; }

    CarRacingEnv(double track_len = racing_cfg::TRACK_LEN, double track_width = racing_cfg::TRACK_WIDTH, double rock_mean = racing_cfg::ROCK_MEAN, double time_budget = racing_cfg::TIME_LIMIT)
        : _len(track_len), _w(track_width), _rock_mean(rock_mean), _T(time_budget) {}

    void reset(State& s, std::uint64_t seed = 0) const {
        s = State{};

        if (seed) s.rng = seed | 1ull;

        for (int i = 0; i < racing_cfg::N_AGENTS; ++i) {
            Car& c = s.cars[i];
            c.s = 0.0;
            c.d = (i - (racing_cfg::N_AGENTS - 1) * 0.5) * (_w * 0.4);
            c.vs = 2.0;
            c.dec_ver = 0;
            c.alive = true;
        }

        s.s_spawn_frontier = 0.0;
        s.rocks.clear();

        s.push(_T, CarRaceEventKind::Timeout);
        s.push(s.expo(_rock_mean), CarRaceEventKind::Spawn);
        for (int i = 0; i < racing_cfg::N_AGENTS; ++i) s.push(0.0, CarRaceEventKind::Decision, i, 0);

        s.cur_agent = pump(const_cast<State&>(s));
        for (auto& c : s.cars) c.rew_acc = 0.0;
    }

    StepResult<Rew> step(State& s, const double* act_in) const {
        if (!s.pending_terminal.empty()) {
            int i = s.pending_terminal.back(); s.pending_terminal.pop_back();
            Car& c = s.cars[i];

            double r = c.rew_acc; c.rew_acc = 0.0; c.closed = true;
            s.cur_agent = s.pending_terminal.empty() ? -1 : s.pending_terminal.back();
            return finish(s, i, r, true, false);
        }

        const int i = s.cur_agent;

        Car& c = s.cars[i];
        c.thrust_s = std::clamp(act_in[0], -1.0, 1.0);
        c.thrust_d = std::clamp(act_in[1], -1.0, 1.0);
        const double commit = act_in[2];

        ++c.dec_ver; ++c.n_dec;
        c.rew_acc -= _dec_cost;

        s.push(s.clock.get() + commit_to_horizon(commit), CarRaceEventKind::ControlExpiry, i, c.dec_ver);

        int hit = 0;
        int result = run(s, i, hit);

        bool terminated = (hit != 0);
        if (terminated) s.cars[i].closed = terminated;
        bool truncated = (!terminated && result == R_TIMEOUT);

        double reward = c.rew_acc; c.rew_acc = 0.0;

        return finish(s, i, reward, terminated, truncated);
    }

    int current_agent(const State& s) const { return s.cur_agent; }

    void observe(const State& s, double* obs_out, int i) const {
        const Car& me = s.cars[i];

        obs_out[0]=me.d;
        obs_out[1]=me.vs;
        obs_out[2]=me.vd;
        obs_out[3]=curvature(me.s);

        const Car& op = s.cars[(i + 1) % racing_cfg::N_AGENTS];

        obs_out[4]=op.d;
        obs_out[5]=op.s - me.s;
        obs_out[6]=op.vs - me.vs;

        std::vector<const Rock*> ahead;
        for (const auto& r : s.rocks) if (r.s >= me.s - 1.0) ahead.push_back(&r);

        std::sort(ahead.begin(), ahead.end(), [&](const Rock* a, const Rock* b){ return std::abs(a->s - me.s) < std::abs(b->s - me.s); });
        for (int k = 0; k < racing_cfg::K_ROCK; ++k) {
            const int idx = 7+2*k;
            if (k < (int)ahead.size()) {
                obs_out[idx]=ahead[k]->s - me.s;
                obs_out[idx+1]=ahead[k]->d - me.d;
            } else {
                obs_out[idx]=0.0;
                obs_out[idx+1]=0.0;
            }
        }
    }

    py::tuple render_state(const RaceState& s) const {
        py::array_t<double> cars({(py::ssize_t)racing_cfg::N_AGENTS, (py::ssize_t)4});
        auto c = cars.mutable_unchecked<2>();
        for (int i = 0; i < racing_cfg::N_AGENTS; ++i) {
            c(i,0) = s.cars[i].s; c(i,1) = s.cars[i].d;
            c(i,2) = s.cars[i].vs; c(i,3) = s.cars[i].alive ? 1.0 : 0.0;
        }

        py::ssize_t M = s.rocks.size();
        py::array_t<double> rocks({M, (py::ssize_t)2});
        auto r = rocks.mutable_unchecked<2>();
        for (py::ssize_t k = 0; k < M; ++k) { r(k,0)=s.rocks[k].s; r(k,1)=s.rocks[k].d; }
        return py::make_tuple(cars, rocks);
    }

    py::dict config() const {
        py::dict d;
        d["track_len"]   = _len;
        d["track_width"] = _w;
        d["rock_r"]      = _rock_r;
        d["car_r"]       = _car_r;
        d["n_agents"]    = num_agents();
        d["time_limit"]  = _T;

        d["h_min"]       = _h_min;
        d["h_max"]       = _h_max;
        d["dec_cost"]    = _dec_cost;
        return d;
    }

private:
    static constexpr int R_RUNNING = 0, R_DECIDE = 1, R_TIMEOUT = 2;

    double commit_to_horizon(double commit) const {
        return _h_min + (std::tanh(commit) + 1.0) * 0.5 * (_h_max - _h_min);
    }

    void queue_terminals(State& s) const {
        for (int i = 0; i < racing_cfg::N_AGENTS; ++i) if (!s.cars[i].closed) s.pending_terminal.push_back(i);
    }

    double curvature(double) const { return 0.0; }

    int pump(State& s) const {
        double dummy = 0.0; int hit = 0;
        run(s, -1, hit);
        return s.cur_agent;
    }

    int run(State& s, int focus, int& hit) const {
        while (!s.q.empty()) {
            bool any = false;
            for (auto& c : s.cars) any |= c.alive;
            if (!any) { queue_terminals(s); return R_TIMEOUT; }

            RaceEvent e = s.q.top(); s.q.pop();

            integrate(s, e.time, focus, hit);

            switch (e.kind) {
                case CarRaceEventKind::Timeout: {
                    queue_terminals(s);
                    return R_TIMEOUT;
                }
                case CarRaceEventKind::Spawn: {
                    spawn_rock(s);
                    s.push(s.clock.get() + s.expo(_rock_mean), CarRaceEventKind::Spawn);
                    break;
                }
                case CarRaceEventKind::ControlExpiry:
                case CarRaceEventKind::Decision: {
                    if (e.agent < 0) break;
                    Car& c = s.cars[e.agent];
                    if (!c.alive) break;
                    if (e.kind == CarRaceEventKind::ControlExpiry && e.ver != c.dec_ver) break;
                    s.cur_agent = e.agent;
                    if (focus >= 0 && hit != 0) return R_RUNNING;
                    return R_DECIDE;
                }
            }

            if (focus >= 0 && hit != 0) return R_RUNNING;
        }
        return R_TIMEOUT;
    }

    void integrate(State& s, double t_target, int focus, int& hit) const {
        constexpr double H = 0.02;
        while (s.clock.get() < t_target - 1e-9) {
            const double h = std::min(H, t_target - s.clock.get());

            for (int i = 0; i < racing_cfg::N_AGENTS; ++i) {
                Car& c = s.cars[i];
                if (!c.alive) continue;
                c.vs += (_a_s * c.thrust_s - _drag * c.vs) * h;
                c.vd += (_a_d * c.thrust_d - _drag_d * c.vd) * h;
                c.s += c.vs * h;
                c.d += c.vd * h;
                
                c.rew_acc += _prog_k * c.vs * h;
                c.rew_acc -= _time_cost * h;

                if (std::abs(c.d) > _w * 0.5) { crash(s, i, focus, hit); }
                else if (c.s >= _len) { c.rew_acc += 100.0; c.alive = false; if (i == focus) hit = 1; }
                else for (const auto& r : s.rocks) {
                    double dds = c.s - r.s, ddd = c.d - r.d;
                    if (dds*dds + ddd*ddd < _rock_r*_rock_r) { crash(s, i, focus, hit); break; }
                }
            }

            for (int i = 0; i < racing_cfg::N_AGENTS; ++i) {
                for (int j = i + 1; j < racing_cfg::N_AGENTS; ++j) {
                    Car& a = s.cars[i]; Car& b = s.cars[j];
                    if (!a.alive || !b.alive) continue;
                    
                    double ds = a.s - b.s, dd = a.d - b.d;
                    double dist_ab = ds*ds + dd*dd;
                    if (dist_ab < _car_r*_car_r && dist_ab > 1e-9) {
                        double dist = std::sqrt(dist_ab);
                        double nx = ds / dist, ny = dd / dist;
                        double push = (_car_r - dist) * 0.5;
                        a.s += nx * push; a.d += ny * push;
                        b.s -= nx * push; b.d -= ny * push;
                        a.vs -= nx * _bump; a.vd -= ny * _bump;
                        b.vs += nx * _bump; b.vd += ny * _bump;
                        a.rew_acc -= _bump_pen;
                        b.rew_acc -= _bump_pen;
                    }
                }
            }

            s.clock.add(h);
            if (focus >= 0 && hit != 0) return;
        }
    }

    void crash(State& s, int i, int focus, int& hit) const {
        s.cars[i].alive = false;
        s.cars[i].rew_acc -= 1.0;
        if (i == focus) hit = -1;
    }

    void spawn_rock(State& s) const {
        double lead = 0.0;
        for (const auto& c : s.cars) lead = std::max(lead, c.s);

        double so = std::max(lead + _spawn_ahead, s.s_spawn_frontier + 0.5);
        if (so >= _len) return;

        double do_ = (s.u01() - 0.5) * (_w - 2 * _rock_r);
        s.rocks.push_back(Rock{ so, do_ });
        s.s_spawn_frontier = so;
    }

    StepResult<Rew> finish(State& s, int i, double reward, bool terminated, bool truncated) const {
        const double tau = s.clock.get() - s.t_last[i];
        s.t_last[i] = s.clock.get();
        
        int obs_agent = (s.cur_agent >= 0) ? s.cur_agent : i;
        return StepResult<Rew>{reward, terminated, truncated, tau};
    }

    double _len, _w, _rock_mean, _T;

    double _drag = 0.4;     // longitudinal drag
    double _drag_d = 1.2;   // lateral drag (grip: car doesn't slide freely)
    double _a_s = 6.0;      // gas authority
    double _a_d = 4.0;      // steer authority
    double _h_min = 0.15;
    double _h_max = 1.2;

    double _rock_r = 0.4;
    double _car_r = 0.6;
    double _bump = 0.5;     // velocity exchanged on car-car contact
    double _bump_pen = 0.3; // reward hit for bumping

    double _spawn_ahead = 12.0;

    double _prog_k = 1.0;   // per unit of forward progress
    double _dec_cost = 0.1;

    double _time_cost = 0.005;
};

static_assert(MultiAgentEnvironment<CarRacingEnv>);