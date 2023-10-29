#include "../pybind_utils.hpp"
#include "denoisors/double_window_filter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(double_window_filter, m) {
	using pybind11::operator""_a;
    
    using DoubleWindowFilter = dv::noise::DoubleWindowFilter<kit::EventStorage>;
    py::class_<DoubleWindowFilter>(m, "init")
        .def(py::init<const cv::Size &, const size_t, const size_t, const size_t>(),
             "resolution"_a, "bufferSize"_a = 36, "searchRadius"_a = 9, "intThreshold"_a = 1)
        .def("accept", &DoubleWindowFilter::accept, "events"_a)
        .def("generateEvents", &DoubleWindowFilter::generateEvents);
}
