#include "../pybind_utils.hpp"
#include "denoisors/reclusive_event_denoisor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(reclusive_event_denoisor, m) {
	using pybind11::operator""_a;
    
    using ReclusiveEventDenoisor = dv::noise::ReclusiveEventDenoisor<kit::EventStorage>;
    py::class_<ReclusiveEventDenoisor>(m, "init")
        .def(py::init<const cv::Size &, const float_t, const int16_t, const float_t, const float_t>(),
             "resolution"_a, "sigmaS"_a = 0.7, "sigmaT"_a = 1, "samplarT"_a = -0.8, "floatThreshold"_a = 1)
        .def("accept", &ReclusiveEventDenoisor::accept, "events"_a)
        .def("generateEvents", &ReclusiveEventDenoisor::generateEvents);
}
