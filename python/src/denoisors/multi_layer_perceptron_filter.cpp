#include "../pybind_utils.hpp"
#include "denoisors/multi_layer_perceptron_filter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(multi_layer_perceptron_filter, m) {
	using pybind11::operator""_a;

    py::implicitly_convertible<std::string, std::filesystem::path>();
    
    using MultiLayerPerceptronFilter = dv::noise::MultiLayerPerceptronFilter<kit::EventStorage>;
    py::class_<MultiLayerPerceptronFilter>(m, "init")
        .def(py::init<const cv::Size &, const fs::path, const size_t, const dv::Duration, const double, const size_t>(),
             "resolution"_a, "modelPath"_a, "batchSize"_a = 5000, "duration"_a = dv::Duration(100000), "floatThreshold"_a = 0.8, "deviceId"_a = 0)
        .def("accept", &MultiLayerPerceptronFilter::accept, "events"_a)
        .def("generateEvents", &MultiLayerPerceptronFilter::generateEvents);
}
