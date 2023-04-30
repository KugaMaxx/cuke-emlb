#include "template.hpp"

namespace kpy {

    class Template : public edn::Template {
    public:
        Template(int16_t sizeX_, int16_t sizeY_) {
            sizeX    = sizeX_;
            sizeY    = sizeY_;
            _LENGTH_ = sizeX * sizeY;
        };

        ~Template() {}

        py::array_t<bool> run(const kore::EventPybind &input, const float_t params_) {
            params    = params_;

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

PYBIND11_MODULE(template, m) {
    py::class_<kpy::Template>(m, "init", py::module_local())
        .def(py::init<int16_t, int16_t>())
        .def("run", &kpy::Template::run, py::arg("input"), py::arg("params") = 1.0);
}
