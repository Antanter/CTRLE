
#pragma once

#include <concepts>
#include <cstddef>
#include <queue>

template<typename T>
concept Event = requires (T t) {
    { t.time } -> std::convertible_to<double>;

    // { t.priority } -> std::convertible_to<double>;
    { t.seq } -> std::convertible_to<size_t>;
    { t.ver } -> std::convertible_to<size_t>;

    t.kind;
} && std::is_scoped_enum_v<decltype(std::declval<T>().kind)>;;

template<Event T>
struct EventCmp {
    bool operator()(const T& a, const T& b) const {
        if (a.time != b.time) return a.time > b.time;
        return a.seq > b.seq;
    }
};

template<Event E>
using EventQueue = std::priority_queue<E, std::vector<E>, EventCmp<E>>;
