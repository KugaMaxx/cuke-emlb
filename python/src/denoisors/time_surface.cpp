#include "../pybind_utils.hpp"
#include "denoisors/time_surface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(time_surface, m) {
	using pybind11::operator""_a;
    
    using TimeSurface = dv::noise::TimeSurface<kit::EventStorage>;
    py::class_<TimeSurface>(m, "init")
        .def(py::init<const cv::Size &, const dv::Duration, const size_t, const double>(),
             "resolution"_a, "duration"_a = dv::Duration(20000), "searchRadius"_a = 1, "floatThreshold"_a = 0.2)
        .def("accept", &TimeSurface::accept, "events"_a)
        .def("generateEvents", &TimeSurface::generateEvents);
}
