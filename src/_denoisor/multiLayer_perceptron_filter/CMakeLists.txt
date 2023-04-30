# Convert module to python package
pybind11_add_module(multiLayer_perceptron_filter _module_py.cpp)
target_link_libraries(multiLayer_perceptron_filter PUBLIC ${TORCH_LIBRARIES} pybind11::module)
