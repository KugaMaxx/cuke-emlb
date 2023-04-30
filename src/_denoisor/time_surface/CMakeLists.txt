# Compile module to dv runtime
if (dv_FOUND)
    add_library(m_time_surface SHARED _module_dv.cpp)
    set_target_properties(m_time_surface PROPERTIES PREFIX "K")
    target_link_libraries(m_time_surface PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
    install(TARGETS m_time_surface DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

# Convert module to python package
pybind11_add_module(time_surface _module_py.cpp)
target_link_libraries(time_surface PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)