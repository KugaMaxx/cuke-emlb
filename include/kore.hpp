#ifndef KORE_H
#define KORE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace kore {

    struct Event {
        int64_t timestamp_;
        int16_t y_;
        int16_t x_;
        bool polarity_;

        Event() : timestamp_(0), y_(0), x_(0), polarity_(false) {}
        Event(const int64_t timestamp, const int16_t y, const int16_t x, const int8_t polarity) : timestamp_(timestamp), y_(y), x_(x), polarity_(polarity) {}

        int64_t timestamp() const {
            return timestamp_;
        }

        int16_t y() const {
            return y_;
        }

        int16_t x() const {
            return x_;
        }

        bool polarity() const {
            return polarity_;
        }
    };

    using EventPybind = py::array_t<Event>;
    using EventPacket = std::vector<Event>;

}

#endif
