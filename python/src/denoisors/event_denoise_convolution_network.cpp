#include "../pybind_utils.hpp"
#include "denoisors/event_denoise_convolution_network.hpp"

namespace py = pybind11;

PYBIND11_MODULE(event_denoise_convolution_network, m) {
	using pybind11::operator""_a;

    py::implicitly_convertible<std::string, std::filesystem::path>();
    
    using EventDenoiseConvolutionNetwork = dv::noise::EventDenoiseConvolutionNetwork<kit::EventStorage>;
    py::class_<EventDenoiseConvolutionNetwork>(m, "init")
        .def(py::init<const cv::Size &, const fs::path, const size_t, const double, const size_t>(),
             "resolution"_a, "modelPath"_a, "batchSize"_a = 1000, "floatThreshold"_a = 0.5, "deviceId"_a = 0)
        .def("accept", &EventDenoiseConvolutionNetwork::accept, "events"_a)
        .def("generateEvents", &EventDenoiseConvolutionNetwork::generateEvents);
}
