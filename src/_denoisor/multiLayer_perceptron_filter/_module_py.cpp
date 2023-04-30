#include "multiLayer_perceptron_filter.hpp"

namespace kpy {

    class MultiLayerPerceptronFilter : public edn::MultiLayerPerceptronFilter {
    public:
        MultiLayerPerceptronFilter(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = sizeX_;
            sizeY    = sizeY_;
            _LENGTH_ = sizeX * sizeY;
        };

        ~MultiLayerPerceptronFilter() {}

        py::array_t<bool> run(const kore::EventPybind &input, const std::string modelPath_, const int32_t batchSize_, const float_t decay_, const int16_t squareR_, const float_t threshold_) {
            modelPath   = modelPath_;
            _BATCHSIZE_ = batchSize_;
            decay       = decay_;
            squareR     = squareR_;
            threshold   = threshold_;

            py::buffer_info buf = input.request();
            kore::Event *ptr    = static_cast<kore::Event *>(buf.ptr);
            kore::EventPacket inEvent(buf.size);
            for (size_t i = 0; i < buf.size; i++) {
                inEvent[i] = ptr[i];
            }

            regenerateParam();

            std::vector<bool> vec;
            vec.reserve(inEvent.size());
            for (auto &evt : inEvent) {
                int k = index++ % _BATCHSIZE_;

                batch[k] = torch::from_blob(buildInput(evt.x(), evt.y(), evt.timestamp(), evt.polarity()).data(), {_ARRAY_}, torch::kFloat);

                // obtain output
                if (k == _BATCHSIZE_ - 1 || index == inEvent.size()) {
                    torch::Tensor output = module.forward({batch.to(device)}).toTensor().to(torch::kCPU);

                    for (; k >= 0; k--) {
                        if (output[k].item<float_t>() >= threshold) {
                            vec.push_back(true);
                        } else {
                            vec.push_back(false);
                        }
                    }
                }
            }

            return py::cast(vec);
        };
    };

}

PYBIND11_MODULE(multiLayer_perceptron_filter, m) {
    py::class_<kpy::MultiLayerPerceptronFilter>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::MultiLayerPerceptronFilter::run, py::arg("input"), py ::arg("model_path") = "None", py::arg("batch_size") = 5000, py ::arg("decay") = 100000, py::arg("square_r") = 3, py::arg("threshold") = 0.8);
}
