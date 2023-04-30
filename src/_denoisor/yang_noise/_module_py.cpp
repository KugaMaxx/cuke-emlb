#include "yang_noise.hpp"

namespace kpy {

    class YangNoise : public edn::YangNoise {
    public:
        YangNoise(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = sizeX_;
            sizeY    = sizeY_;
            _LENGTH_ = sizeX * sizeY;
        };

        ~YangNoise() {}

        py::array_t<bool> run(const kore::EventPybind &input, const float_t deltaT_, const float_t squareR_, const int16_t threshold_) {
            deltaT = deltaT_;
            squareR = squareR_;
            threshold = threshold_;

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
                bool isNoise = calculateDensity(evt.x(), evt.y(), evt.timestamp(), evt.polarity());

                if (isNoise) {
                    vec.push_back(true);
                } else {
                    vec.push_back(false);
                }
            }

            return py::cast(vec);
        };
    };

}

PYBIND11_MODULE(yang_noise, m) {
    py::class_<kpy::YangNoise>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::YangNoise::run, py::arg("input"), py::arg("delta_t") = 10000, py::arg("square_r") = 1, py::arg("threshold") = 2);
}
