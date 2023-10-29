#include "../pybind_utils.hpp"
#include "denoisors/event_flow.hpp"

namespace py = pybind11;

PYBIND11_MODULE(event_flow, m) {
	using pybind11::operator""_a;

    using EventFlow = dv::noise::EventFlow<kit::EventStorage>;
    py::class_<EventFlow>(m, "init")
        .def(py::init<const cv::Size &, const dv::Duration, const size_t, const double>(),
             "resolution"_a, "duration"_a = dv::Duration(2000), "searchRadius"_a = 1, "floatThreshold"_a = 20.0)
        .def("accept", &EventFlow::accept, "events"_a)
        .def("generateEvents", &EventFlow::generateEvents);
}
