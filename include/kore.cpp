#include "kore.hpp"

void hello() {
    std::cout << "welcome   --KugaMaxx@outlook.com" << std::endl;
    return;
}

PYBIND11_MODULE(kore, m)
{
    PYBIND11_NUMPY_DTYPE(kore::Event, timestamp_, y_, x_, polarity_);
    m.def("hello", &hello, "say hello");
}
