#include "../pybind_utils.hpp"
#include "denoisors/yang_noise.hpp"

namespace py = pybind11;

PYBIND11_MODULE(yang_noise, m) {
	using pybind11::operator""_a;
    
    using YangNoise = dv::noise::YangNoise<kit::EventStorage>;
    py::class_<YangNoise>(m, "init")
        .def(py::init<const cv::Size &, const dv::Duration, const size_t, const size_t>(),
             "resolution"_a, "duration"_a = dv::Duration(2000), "searchRadius"_a = 1, "intThreshold"_a = 1)
        .def("accept", &YangNoise::accept, "events"_a)
        .def("generateEvents", &YangNoise::generateEvents);
}
