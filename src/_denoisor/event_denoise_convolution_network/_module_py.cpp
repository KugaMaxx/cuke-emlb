#include "event_denoise_convolution_network.hpp"

namespace kpy {

    class EventDenoiseConvolutionNetwork : public edn::EventDenoiseConvolutionNetwork {
    public:
        EventDenoiseConvolutionNetwork(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = sizeX_;
            sizeY    = sizeY_;
            _LENGTH_ = sizeX * sizeY;
        };

        ~EventDenoiseConvolutionNetwork() {}

        py::array_t<bool> run(const kore::EventPybind &input, const std::string modelPath_, const int32_t batchSize_, const int32_t depth_, const int16_t squareR_, const float_t threshold_) {
            modelPath   = modelPath_;
            _BATCHSIZE_ = batchSize_;
            _DEPTH_     = depth_;
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

                batch[k] = torch::from_blob(buildInput(evt.x(), evt.y(), evt.timestamp(), evt.polarity()).data(), {2 * _DEPTH_, _SQUARE_L_, _SQUARE_L_}, torch::kFloat);

                // obtain output
                if (k == _BATCHSIZE_ - 1 || index == inEvent.size()) {
                    torch::Tensor output = module.forward({batch.to(device)}).toTensor().to(torch::kCPU);

                    for (; k >= 0; k--) {
                        std::cout << output[k] << std::endl;
                        if (output[k][1].item<float_t>() >= threshold) {
                            vec.push_back(true);
                        } else {
                            vec.push_back(false);
                        }
                    }
                    break;
                }
            }

            return py::cast(vec);
        };
    };

}

PYBIND11_MODULE(event_denoise_convolution_network, m) {
    py::class_<kpy::EventDenoiseConvolutionNetwork>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::EventDenoiseConvolutionNetwork::run, py::arg("input"), py ::arg("model_path") = "None", py::arg("batch_size") = 1000, py ::arg("depth") = 2, py::arg("square_r") = 12, py::arg("threshold") = 0.8);
}
