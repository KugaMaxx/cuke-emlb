# Compile module to dv runtime
if (dv_FOUND)
    add_library(m_yang_noise SHARED _module_dv.cpp)
    set_target_properties(m_yang_noise PROPERTIES PREFIX "K")
    target_link_libraries(m_yang_noise PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
    install(TARGETS m_yang_noise DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

# Convert module to python package
pybind11_add_module(yang_noise _module_py.cpp)
target_link_libraries(yang_noise PUBLIC dv::sdk pybind11::module)
