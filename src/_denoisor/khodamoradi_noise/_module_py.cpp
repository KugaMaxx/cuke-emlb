#include "khodamoradi_noise.hpp"

namespace kpy {

    class KhodamoradiNoise : public edn::KhodamoradiNoise {
    public:
        KhodamoradiNoise(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = sizeX_;
            sizeY    = sizeY_;
            _LENGTH_ = sizeX * sizeY;
        };

        ~KhodamoradiNoise() {}

        py::array_t<bool> run(const kore::EventPybind &input, const int32_t deltaT_, const int16_t threshold_) {
            deltaT    = deltaT_;
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

PYBIND11_MODULE(khodamoradi_noise, m) {
    py::class_<kpy::KhodamoradiNoise>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::KhodamoradiNoise::run, py::arg("input"), py::arg("delta_t") = 1000, py::arg("threshold") = 1);
}
