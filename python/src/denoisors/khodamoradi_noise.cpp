#include "../pybind_utils.hpp"
#include "denoisors/khodamoradi_noise.hpp"

namespace py = pybind11;

PYBIND11_MODULE(khodamoradi_noise, m) {
	using pybind11::operator""_a;
    
    using KhodamoradiNoise = dv::noise::KhodamoradiNoise<kit::EventStorage>;
    py::class_<KhodamoradiNoise>(m, "init")
        .def(py::init<const cv::Size &, const dv::Duration, const size_t>(),
             "resolution"_a, "duration"_a = dv::Duration(2000), "intThreshold"_a = 1)
        .def("accept", &KhodamoradiNoise::accept, "events"_a)
        .def("generateEvents", &KhodamoradiNoise::generateEvents);
}
