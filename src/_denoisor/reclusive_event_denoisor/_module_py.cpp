#include "reclusive_event_denoisor.hpp"

namespace kpy {

    class ReclusiveEventDenoisor : public edn::ReclusiveEventDenoisor {
    public:
        ReclusiveEventDenoisor(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = ceil((float)sizeX_ / _THREAD_) * _THREAD_;
            sizeY    = ceil((float)sizeY_ / _THREAD_) * _THREAD_;
            _LENGTH_ = sizeX * sizeY;

            Xt = (float_t *)calloc(_POLES_ * _LENGTH_, sizeof(float_t));
            Yt = (float_t *)calloc(1 * _LENGTH_, sizeof(float_t));
            Ut = (float_t *)calloc(1 * _LENGTH_, sizeof(float_t));
        };

        ~ReclusiveEventDenoisor() {
            free(Xt);
            free(Yt);
            free(Ut);
        }

        py::array_t<bool> run(const kore::EventPybind &input, const float_t sigmaS_, const int16_t sigmaT_, const float_t samplarT_, const float_t threshold_) {
            sigmaS    = sigmaS_;
            sigmaT    = sigmaT_;
            samplarT  = samplarT_;
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

PYBIND11_MODULE(reclusive_event_denoisor, m) {
    py::class_<kpy::ReclusiveEventDenoisor>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::ReclusiveEventDenoisor::run, py::arg("input"), py::arg("sigmaS") = 0.7, py::arg("sigmaT") = 1, py::arg("samplarT") = -0.8, py::arg("threshold") = 0.2);
}
