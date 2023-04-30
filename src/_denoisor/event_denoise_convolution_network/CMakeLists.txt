# Convert module to python package
pybind11_add_module(event_denoise_convolution_network _module_py.cpp)
target_link_libraries(event_denoise_convolution_network PUBLIC ${TORCH_LIBRARIES} pybind11::module)
