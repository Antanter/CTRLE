#pragma once

#include <concepts>
#include <ranges>

// consider all posibilities - numbers + vectors of numbers. vectors of vectors are not included, since reserved for later inprovements.
template<typename T>
concept Num = std::integral<T> || std::floating_point<T>;
template<typename T>
concept NumRange = std::ranges::range<T> && Num<std::ranges::range_value_t<T>>;
template<typename T>
concept NumLike = Num<T> || NumRange<T>;

template <typename Rew>
struct StepResult {
    Rew rew;
    bool terminated;
    bool truncated;
    double tau;
};