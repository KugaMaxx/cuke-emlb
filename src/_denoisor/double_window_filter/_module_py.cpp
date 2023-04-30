#include "double_window_filter.hpp"

namespace kpy {

    class DoubleWindowFilter : public edn::DoubleWindowFilter {
    public:
        DoubleWindowFilter(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = sizeX_;
            sizeY    = sizeY_;
            _LENGTH_ = sizeX * sizeY;
        };

        ~DoubleWindowFilter() {}

        py::array_t<bool> run(const kore::EventPybind &input, const uint8_t wLen_, const int16_t squareR_, const int16_t threshold_) {
            squareR   = squareR_;
            threshold = threshold_;
            wLen = wLen_;

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

PYBIND11_MODULE(double_window_filter, m) {
    py::class_<kpy::DoubleWindowFilter>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::DoubleWindowFilter::run, py::arg("input"), py::arg("w_len") = 36, py ::arg("square_r") = 9, py::arg("threshold") = 1);
}
