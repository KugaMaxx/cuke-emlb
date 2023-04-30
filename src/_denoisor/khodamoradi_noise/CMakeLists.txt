# Compile module to dv runtime
if (dv_FOUND)
    add_library(m_khodamoradi_noise SHARED _module_dv.cpp)
    set_target_properties(m_khodamoradi_noise PROPERTIES PREFIX "K")
    target_link_libraries(m_khodamoradi_noise PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
    install(TARGETS m_khodamoradi_noise DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

# Convert module to python package
pybind11_add_module(khodamoradi_noise _module_py.cpp)
target_link_libraries(khodamoradi_noise PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)