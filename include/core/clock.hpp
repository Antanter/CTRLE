
#pragma once

class Clock {
    double time = 0.0;

public:
    void add(double _delta) { time += _delta; }
    void set(double _time) { time = _time; }
    double get() const { return time; }

    void reset() { set(0.0); }
};