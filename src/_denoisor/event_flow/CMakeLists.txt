# Compile module to dv runtime
if (dv_FOUND)
    add_library(m_event_flow SHARED _module_dv.cpp)
    set_target_properties(m_event_flow PROPERTIES PREFIX "K")
    target_link_libraries(m_event_flow PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
    install(TARGETS m_event_flow DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

# Convert module to python package
pybind11_add_module(event_flow _module_py.cpp)
target_link_libraries(event_flow PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)