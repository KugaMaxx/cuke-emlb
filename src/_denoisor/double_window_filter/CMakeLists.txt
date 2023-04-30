# Compile module to dv runtime
if (dv_FOUND)
    add_library(m_double_window_filter SHARED _module_dv.cpp)
    set_target_properties(m_double_window_filter PROPERTIES PREFIX "K")
    target_link_libraries(m_double_window_filter PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
    install(TARGETS m_double_window_filter DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

# Convert module to python package
pybind11_add_module(double_window_filter _module_py.cpp)
target_link_libraries(double_window_filter PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
