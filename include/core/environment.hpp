
#pragma once

#include "core/type.hpp"
#include <concepts>

template<typename T>
concept State = requires (T t) {
    { t.push() };
};

template<typename T>
concept Obs = NumLike<T>;

template<typename T>
concept Act = NumLike<T>;

template<typename T>
concept Rew = NumLike<T>;

template<typename T>
concept Environment = requires (T t, const T ct, typename T::State state, const double* act_in, double* obs_out) {
    typename T::State;
    typename T::Act;
    typename T::Obs;
    typename T::Rew;

    requires Act<typename T::Act>;
    requires Obs<typename T::Obs>;
    requires Rew<typename T::Rew>;

    { ct.obs_dim() } -> NumLike;
    { ct.act_dim() } -> NumLike;

    { t.reset(state) };
    { t.step(state, act_in) } -> std::same_as<StepResult<typename T::Rew>>;
    { ct.observe(state, obs_out) };
};

template<typename T>
concept MultiAgentEnvironment = requires (T t, const T ct, typename T::State state, const double* act_in, double* obs_out, int agent_id) {
    typename T::State;
    typename T::Act;
    typename T::Obs;
    typename T::Rew;

    requires Act<typename T::Act>;
    requires Obs<typename T::Obs>;
    requires Rew<typename T::Rew>;

    { ct.num_agents() } -> NumLike;
    { ct.obs_dim() } -> NumLike;
    { ct.act_dim() } -> NumLike;

    { t.reset(state) };
    { t.step(state, act_in) } -> std::same_as<StepResult<typename T::Rew>>;
    { ct.observe(state, obs_out, agent_id) };
};
